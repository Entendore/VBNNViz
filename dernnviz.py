import sys
import uuid
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QTextEdit, QGroupBox, QListWidget,
    QSplitter, QComboBox, QFormLayout
)
from PySide6.QtCore import Qt, Signal
import chromadb

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class NeuralNetVectorizer:
    def __init__(self, input_size=4, hidden_size=8, output_size=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1_size = input_size * hidden_size       # 32
        self.b1_size = hidden_size                     # 8
        self.w2_size = hidden_size * output_size       # 16  (FIXED: was input_size * output_size)
        self.b2_size = output_size                     # 2
        self.vector_size = self.w1_size + self.b1_size + self.w2_size + self.b2_size  # 58

        # ── Sample points for derivative evaluation ──────────────────────
        # These span the input space so derivatives capture global behavior
        self.sample_points = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([-1.0, -1.0, -1.0, -1.0]),
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([1.0, -1.0, 1.0, -1.0]),
            np.array([-1.0, 1.0, -1.0, 1.0]),
            np.array([0.5, -0.5, 0.5, -0.5]),
        ]

    # ═══════════════════════════════════════════════════════════════════
    #  DERIVATIVE COMPUTATION  (1st, 2nd, 3rd, 4th)
    # ═══════════════════════════════════════════════════════════════════

    def compute_derivatives(self, vector, x, h=0.1):
        """
        Compute 1st, 2nd, 3rd, 4th derivatives of each output w.r.t. each
        input dimension using central finite differences.

        1st deriv  — sensitivity / slope      (how responsive is the net?)
        2nd deriv  — curvature / acceleration (how fast does sensitivity change?)
        3rd deriv  — jerk                     (rate of change of curvature)
        4th deriv  — snap                     (rate of change of jerk)

        For each sample point x (4 inputs), we perturb each input dimension
        and record how the 2 outputs change. This yields:
           d1: 4×2 = 8 values,  d2: 8 values,  d3: 8 values,  d4: 8 values
        per sample point.
        """
        x = np.array(x, dtype=float)
        base = self.forward_pass(vector, x)
        n_inputs = len(x)

        d1_all, d2_all, d3_all, d4_all = [], [], [], []

        for i in range(n_inputs):
            # 5-point stencil perturbations
            x_p2 = x.copy(); x_p2[i] += 2 * h
            x_p1 = x.copy(); x_p1[i] += h
            x_m1 = x.copy(); x_m1[i] -= h
            x_m2 = x.copy(); x_m2[i] -= 2 * h

            f_p2 = self.forward_pass(vector, x_p2)
            f_p1 = self.forward_pass(vector, x_p1)
            f_m1 = self.forward_pass(vector, x_m1)
            f_m2 = self.forward_pass(vector, x_m2)

            # Central finite-difference formulas (O(h²) truncation error)
            d1 = (f_p1 - f_m1) / (2 * h)                              # 1st derivative
            d2 = (f_p1 - 2 * base + f_m1) / (h ** 2)                  # 2nd derivative
            d3 = (f_p2 - 2 * f_p1 + 2 * f_m1 - f_m2) / (2 * h ** 3)   # 3rd derivative
            d4 = (f_p2 - 4 * f_p1 + 6 * base - 4 * f_m1 + f_m2) / (h ** 4)  # 4th derivative

            d1_all.extend(d1.tolist())
            d2_all.extend(d2.tolist())
            d3_all.extend(d3.tolist())
            d4_all.extend(d4.tolist())

        return {'d1': d1_all, 'd2': d2_all, 'd3': d3_all, 'd4': d4_all}

    def compute_derivative_vector(self, vector, normalize=True):
        """
        Evaluate derivatives at ALL sample points and flatten into a single
        feature vector. Each derivative order is independently L2-normalized
        so that no single order dominates the clustering.

        Returns a list of length 7 sample_pts × 4 orders × 8 vals = 224.
        """
        all_d1, all_d2, all_d3, all_d4 = [], [], [], []

        for x in self.sample_points:
            derivs = self.compute_derivatives(vector, x)
            all_d1.extend(derivs['d1'])
            all_d2.extend(derivs['d2'])
            all_d3.extend(derivs['d3'])
            all_d4.extend(derivs['d4'])

        if normalize:
            for arr in [all_d1, all_d2, all_d3, all_d4]:
                n = np.linalg.norm(arr)
                if n > 1e-12:
                    arr /= n  # in-place on the list won't work; fix below
            # Re-do properly with numpy
            d1 = np.array(all_d1); d1 = d1 / (np.linalg.norm(d1) + 1e-8)
            d2 = np.array(all_d2); d2 = d2 / (np.linalg.norm(d2) + 1e-8)
            d3 = np.array(all_d3); d3 = d3 / (np.linalg.norm(d3) + 1e-8)
            d4 = np.array(all_d4); d4 = d4 / (np.linalg.norm(d4) + 1e-8)
            return np.concatenate([d1, d2, d3, d4]).tolist()
        else:
            return all_d1 + all_d2 + all_d3 + all_d4

    def compute_augmented_vector(self, vector, mode='combined'):
        """
        Build a feature vector for clustering/similarity:
          - 'weights_only'       : raw weight vector (58 dims)
          - 'derivatives_only'  : derivative fingerprint (224 dims)
          - 'combined'           : L2-normalized weights ⊕ derivatives (282 dims)
        """
        if mode == 'weights_only':
            return vector
        elif mode == 'derivatives_only':
            return self.compute_derivative_vector(vector)
        elif mode == 'combined':
            deriv_vec = self.compute_derivative_vector(vector)
            w = np.array(vector)
            d = np.array(deriv_vec)
            w_norm = w / (np.linalg.norm(w) + 1e-8)
            d_norm = d / (np.linalg.norm(d) + 1e-8)
            return np.concatenate([w_norm, d_norm]).tolist()
        return vector

    def get_derivative_summary(self, vector):
        """Compact summary of derivative magnitudes at a representative point."""
        x = np.array([0.5, 0.5, 0.5, 0.5])
        derivs = self.compute_derivatives(vector, x)
        summary = {}
        for order in ['d1', 'd2', 'd3', 'd4']:
            vals = np.array(derivs[order])
            summary[order] = {
                'l2_norm': float(np.linalg.norm(vals)),
                'mean_abs': float(np.mean(np.abs(vals))),
                'max_abs': float(np.max(np.abs(vals))),
                'values': vals.tolist(),
            }
        return summary

    # ═══════════════════════════════════════════════════════════════════
    #  NETWORK GENERATION
    # ═══════════════════════════════════════════════════════════════════

    def generate_random_nn(self):
        w1 = np.random.randn(self.input_size, self.hidden_size)
        b1 = np.random.randn(self.hidden_size)
        w2 = np.random.randn(self.hidden_size, self.output_size)  # FIXED
        b2 = np.random.randn(self.output_size)
        vector = np.concatenate([w1.flatten(), b1, w2.flatten(), b2])
        return vector.tolist()

    def generate_bayesian_nn(self, successful_vectors, feature_mode='combined'):
        """
        Multi-Modal Empirical Bayes sampling with derivative-aware clustering.

        Instead of clustering on raw weights alone, we cluster on a feature
        space that includes 1st-4th derivatives. This groups networks by
        BEHAVIORAL similarity (how they respond to inputs) rather than just
        weight similarity, leading to more meaningful offspring.

        After clustering on the augmented feature space, we sample new
        weights from the distribution of the chosen cluster's original
        weight vectors.
        """
        if not successful_vectors:
            return self.generate_random_nn()

        # ── Build feature matrix for clustering ────────────────────────
        if feature_mode == 'weights_only':
            data = np.array(successful_vectors)
        else:
            augmented = [self.compute_augmented_vector(v, feature_mode)
                          for v in successful_vectors]
            data = np.array(augmented)

        # ── Multi-modal clustering ────────────────────────────────────
        if len(data) >= 4:
            n_clusters = min(3, len(data) // 2)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(data)

            chosen_cluster = np.random.randint(0, n_clusters)
            cluster_indices = np.where(labels == chosen_cluster)[0]
        else:
            cluster_indices = np.arange(len(successful_vectors))

        # ── Sample from original weight vectors of chosen cluster ──────
        cluster_weights = np.array([successful_vectors[i] for i in cluster_indices])

        mean_weights = np.mean(cluster_weights, axis=0)
        std_weights = np.std(cluster_weights, axis=0)
        std_weights[std_weights < 1e-5] = 1e-5

        new_vector = np.random.normal(mean_weights, std_weights)
        return new_vector.tolist()

    def unflatten_nn(self, vector):
        v = np.array(vector)
        idx = 0
        w1 = v[idx : idx + self.w1_size].reshape(self.input_size, self.hidden_size); idx += self.w1_size
        b1 = v[idx : idx + self.b1_size]; idx += self.b1_size
        w2 = v[idx : idx + self.w2_size].reshape(self.hidden_size, self.output_size); idx += self.w2_size  # FIXED
        b2 = v[idx : idx + self.b2_size]
        return w1, b1, w2, b2

    def forward_pass(self, vector, x):
        w1, b1, w2, b2 = self.unflatten_nn(vector)
        h = np.maximum(0, np.dot(x, w1) + b1)          # ReLU hidden layer
        o = 1 / (1 + np.exp(-(np.dot(h, w2) + b2)))      # Sigmoid output
        return o

    def evaluate_benchmarks(self, vector):
        results = {}
        out_a = self.forward_pass(vector, np.array([1.0, 1.0, 1.0, 1.0]))
        results['bench_a'] = "pass" if out_a[0] > 0.6 else "fail"

        out_b = self.forward_pass(vector, np.array([0.0, 0.0, 0.0, 0.0]))
        results['bench_b'] = "pass" if out_b[1] < 0.4 else "fail"

        out_c = self.forward_pass(vector, np.array([-1.0, -1.0, 0.0, 0.0]))
        results['bench_c'] = "pass" if out_c[0] > 0.5 else "fail"

        return results


# ═══════════════════════════════════════════════════════════════════════
#  VECTOR SPACE CANVAS  (PCA projection with derivative-aware coloring)
# ═══════════════════════════════════════════════════════════════════════

class VectorSpaceCanvas(FigureCanvas):
    pointClicked = Signal(int)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setParent(parent)

        self.db_coords = []
        self.db_metas = []
        self.mpl_connect('button_press_event', self.on_click)
        self.mpl_connect('motion_notify_event', self.on_hover)

        self.tooltip = self.ax.annotate("", xy=(0, 0), xytext=(10, 10),
                                        textcoords="offset points",
                                        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.9),
                                        arrowprops=dict(arrowstyle="->"))
        self.tooltip.set_visible(False)
        self.draw_empty()

    def draw_empty(self):
        self.ax.clear()
        self.ax.set_title("Vector Space (PCA Projection)")
        self.ax.text(0.5, 0.5, "Database Empty", ha='center', va='center',
                     transform=self.ax.transAxes)
        self.draw()

    def on_click(self, event):
        if event.inaxes != self.ax or not self.db_coords.size:
            return
        click_coord = np.array([event.xdata, event.ydata])
        distances = np.linalg.norm(self.db_coords - click_coord, axis=1)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] < 0.5:
            self.pointClicked.emit(closest_idx)

    def on_hover(self, event):
        if event.inaxes != self.ax or not self.db_coords.size:
            if self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.draw_idle()
            return

        hover_coord = np.array([event.xdata, event.ydata])
        distances = np.linalg.norm(self.db_coords - hover_coord, axis=1)
        closest_idx = np.argmin(distances)

        if distances[closest_idx] < 0.3:
            meta = self.db_metas[closest_idx]
            n_type = meta.get('type', 'Unknown')
            b_a = meta.get('bench_a', 'N/A')[:1].upper()
            b_b = meta.get('bench_b', 'N/A')[:1].upper()
            b_c = meta.get('bench_c', 'N/A')[:1].upper()
            d1 = meta.get('d1_norm', 0)
            d2 = meta.get('d2_norm', 0)

            text = (f"Idx: {closest_idx}\nType: {n_type}\n"
                    f"Bench: A:{b_a} B:{b_b} C:{b_c}\n"
                    f"|d1|={d1:.2f}  |d2|={d2:.2f}")
            self.tooltip.xy = (event.xdata, event.ydata)
            self.tooltip.set_text(text)
            self.tooltip.set_visible(True)
            self.draw_idle()
        else:
            if self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.draw_idle()

    def plot_pca(self, db_vectors, metadatas, query_vector=None,
                 result_indices=None, color_mode="pca_cluster",
                 vectorizer=None):
        """
        Plot the PCA projection of the database.
        When color_mode == 'derivative_cluster', the clustering is based
        on derivative features (1st-4th) instead of raw weights.
        """
        self.ax.clear()
        self.db_metas = metadatas

        if not db_vectors:
            self.db_coords = []
            self.draw_empty()
            return

        # ── PCA projection on raw weights (consistent spatial layout) ─
        all_data = np.array(db_vectors)
        if query_vector is not None:
            all_data = np.vstack([all_data, np.array(query_vector)])

        pca = PCA(n_components=2)
        coords = pca.fit_transform(all_data)

        db_coords = coords[:-1] if query_vector is not None else coords
        self.db_coords = db_coords

        types = [m.get('type', 'Random') for m in metadatas]
        is_bayes = ["bayes" in t.lower() for t in types]
        is_bayes_arr = np.array(is_bayes)

        # ── Determine cluster coloring ────────────────────────────────
        if color_mode == "pca_cluster":
            n_clusters = min(4, len(db_coords))
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            clusters = kmeans.fit_predict(db_coords)

            self.ax.scatter(db_coords[~is_bayes_arr, 0], db_coords[~is_bayes_arr, 1],
                            c=clusters[~is_bayes_arr], cmap='viridis', marker='o',
                            alpha=0.7, label='Random')
            self.ax.scatter(db_coords[is_bayes_arr, 0], db_coords[is_bayes_arr, 1],
                            c=clusters[is_bayes_arr], cmap='viridis', marker='^',
                            alpha=0.9, label='Bayesian', edgecolors='black')
            self.ax.set_title("Weight-Space Clusters (o=Random, ^=Bayes)")

        elif color_mode == "derivative_cluster":
            # ── Cluster on DERIVATIVE features, display on weight PCA ──
            if vectorizer is not None:
                deriv_vectors = [vectorizer.compute_derivative_vector(v) for v in db_vectors]
                deriv_data = np.array(deriv_vectors)

                n_clusters = min(4, len(deriv_data))
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                clusters = kmeans.fit_predict(deriv_data)

                self.ax.scatter(db_coords[~is_bayes_arr, 0], db_coords[~is_bayes_arr, 1],
                                c=clusters[~is_bayes_arr], cmap='plasma', marker='o',
                                alpha=0.7, label='Random')
                self.ax.scatter(db_coords[is_bayes_arr, 0], db_coords[is_bayes_arr, 1],
                                c=clusters[is_bayes_arr], cmap='plasma', marker='^',
                                alpha=0.9, label='Bayesian', edgecolors='black')
                self.ax.set_title("Derivative-Behavior Clusters (d1-d4 fingerprint)")
            else:
                self.ax.scatter(db_coords[:, 0], db_coords[:, 1], c='gray', alpha=0.5)
                self.ax.set_title("Derivative Clusters (no vectorizer)")

        elif color_mode in ["bench_a", "bench_b", "bench_c"]:
            colors = ['green' if m.get(color_mode) == 'pass' else 'red' for m in metadatas]
            self.ax.scatter(db_coords[~is_bayes_arr, 0], db_coords[~is_bayes_arr, 1],
                            c=[c for i, c in enumerate(colors) if not is_bayes[i]],
                            marker='o', alpha=0.7, label='Random')
            self.ax.scatter(db_coords[is_bayes_arr, 0], db_coords[is_bayes_arr, 1],
                            c=[c for i, c in enumerate(colors) if is_bayes[i]],
                            marker='^', alpha=0.9, label='Bayesian', edgecolors='black')
            self.ax.set_title(f"{color_mode.capitalize()} Results (Green=Pass, Red=Fail)")

        if result_indices:
            res_coords = db_coords[result_indices]
            self.ax.scatter(res_coords[:, 0], res_coords[:, 1], c='cyan', s=100,
                            marker='s', label='Search Result', edgecolors='k', zorder=4)

        if query_vector is not None:
            q_coord = coords[-1]
            self.ax.scatter(q_coord[0], q_coord[1], c='magenta', marker='*',
                            s=200, label='Query NN', edgecolors='k', zorder=5)

        self.ax.legend(fontsize=8)
        self.ax.set_xlabel("PC1"); self.ax.set_ylabel("PC2")
        self.draw()


# ═══════════════════════════════════════════════════════════════════════
#  NN ARCHITECTURE CANVAS  (network diagram + derivative profile)
# ═══════════════════════════════════════════════════════════════════════

class NNArchitectureCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 6), tight_layout=True)
        super().__init__(self.fig)
        # Top subplot: network architecture
        self.ax = self.fig.add_subplot(2, 1, 1)
        # Bottom subplot: derivative profile
        self.ax_deriv = self.fig.add_subplot(2, 1, 2)
        self.setParent(parent)
        self.draw_empty()

    def draw_empty(self):
        self.ax.clear()
        self.ax_deriv.clear()
        self.ax.set_title("Neural Network Architecture")
        self.ax.axis('off')
        self.ax.text(0.5, 0.5, "Click a point in the Vector Space\nor Generate a Query NN",
                     ha='center', va='center', transform=self.ax.transAxes)
        self.ax_deriv.set_title("Derivative Profile (d1–d4)")
        self.ax_deriv.axis('off')
        self.draw()

    def draw_nn(self, vector, title="Neural Network", metadata=None, vectorizer=None):
        self.ax.clear()
        self.ax_deriv.clear()
        self.ax.set_title(title, fontsize=10)
        self.ax.axis('off')

        w1, b1, w2, b2 = NeuralNetVectorizer().unflatten_nn(vector)
        x_in, x_hid, x_out = 0.1, 0.5, 0.9
        y_in = np.linspace(0.2, 0.8, 4)
        y_hid = np.linspace(0.1, 0.9, 8)
        y_out = np.linspace(0.35, 0.65, 2)

        # Draw connections
        for i in range(4):
            for h in range(8):
                weight = w1[i, h]
                color = 'blue' if weight > 0 else 'red'
                alpha = min(abs(weight), 1.0)
                self.ax.plot([x_in, x_hid], [y_in[i], y_hid[h]],
                             color=color, alpha=alpha, linewidth=1)
        for h in range(8):
            for o in range(2):
                weight = w2[h, o]
                color = 'blue' if weight > 0 else 'red'
                alpha = min(abs(weight), 1.0)
                self.ax.plot([x_hid, x_out], [y_hid[h], y_out[o]],
                             color=color, alpha=alpha, linewidth=1)

        # Draw nodes (bias = size)
        sizes_hid = [100 + abs(b) * 200 for b in b1]
        sizes_out = [100 + abs(b) * 200 for b in b2]
        self.ax.scatter([x_in] * 4, y_in, s=100, c='black', zorder=5)
        self.ax.scatter([x_hid] * 8, y_hid, s=sizes_hid, c='black', zorder=5)
        self.ax.scatter([x_out] * 2, y_out, s=sizes_out, c='black', zorder=5)

        self.ax.text(x_in, 0.95, 'Input', ha='center')
        self.ax.text(x_hid, 0.95, 'Hidden', ha='center')
        self.ax.text(x_out, 0.95, 'Output', ha='center')
        self.ax.plot([], [], color='blue', label='Positive Weight')
        self.ax.plot([], [], color='red', label='Negative Weight')

        if metadata:
            bench_text = (f"A: {metadata.get('bench_a', 'N/A')[:4].upper()}\n"
                          f"B: {metadata.get('bench_b', 'N/A')[:4].upper()}\n"
                          f"C: {metadata.get('bench_c', 'N/A')[:4].upper()}")
            self.ax.text(1.05, 0.5, bench_text, transform=self.ax.transAxes,
                         fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))

        self.ax.legend(loc='lower left', fontsize=8)

        # ── Derivative Profile subplot ────────────────────────────────
        self._draw_derivative_profile(vector, vectorizer)

        self.draw()

    def _draw_derivative_profile(self, vector, vectorizer=None):
        """Draw a bar chart of derivative L2 norms (d1, d2, d3, d4)."""
        if vectorizer is None:
            vectorizer = NeuralNetVectorizer()

        summary = vectorizer.get_derivative_summary(vector)

        orders = ['d1\n(Slope)', 'd2\n(Curv.)', 'd3\n(Jerk)', 'd4\n(Snap)']
        norms = [summary[o]['l2_norm'] for o in ['d1', 'd2', 'd3', 'd4']]
        mean_abs = [summary[o]['mean_abs'] for o in ['d1', 'd2', 'd3', 'd4']]

        x = np.arange(len(orders))
        width = 0.35

        # Use log scale if values span large range
        display_norms = [max(n, 1e-10) for n in norms]

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
        bars = self.ax_deriv.bar(x, display_norms, width, color=colors, alpha=0.8,
                                 edgecolor='black', linewidth=0.5)

        self.ax_deriv.set_xticks(x)
        self.ax_deriv.set_xticklabels(orders, fontsize=8)
        self.ax_deriv.set_ylabel('L2 Norm (log)', fontsize=9)
        self.ax_deriv.set_title('Derivative Fingerprint @ [0.5, 0.5, 0.5, 0.5]', fontsize=9)
        self.ax_deriv.set_yscale('log')

        # Annotate bars with values
        for bar, val, ma in zip(bars, norms, mean_abs):
            self.ax_deriv.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.3,
                               f'{val:.2f}', ha='center', va='bottom', fontsize=7)

        # Add interpretation text
        interp = ("Higher d1 = more sensitive to inputs\n"
                  "Higher d2 = sharper decision boundaries\n"
                  "Higher d3/d4 = more complex nonlinearity")
        self.ax_deriv.text(0.98, 0.95, interp, transform=self.ax_deriv.transAxes,
                           fontsize=6, ha='right', va='top', style='italic',
                           bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))


# ═══════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Vector DB — Evolution + Derivatives Edition")
        self.resize(1320, 900)

        self.vectorizer = NeuralNetVectorizer()
        self.current_query_vector = None
        self.db_vectors, self.db_ids, self.db_metas = [], [], []
        self.current_result_indices = []

        # Derivative cache
        self._deriv_cache = {}
        self._deriv_cache_valid = False

        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="neural_networks_deriv")

        # ── UI Setup ──────────────────────────────────────────────────
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(360)

        # Group 1: Random Generation
        gen_group = QGroupBox("1. Random Baseline Generation")
        gen_layout = QVBoxLayout()
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 1000)
        self.count_spin.setValue(100)
        self.generate_btn = QPushButton("Generate Random NNs")
        self.generate_btn.clicked.connect(self.generate_and_store)
        self.clear_btn = QPushButton("Clear Database")
        self.clear_btn.setStyleSheet("color: red;")
        self.clear_btn.clicked.connect(self.clear_database)
        gen_layout.addWidget(QLabel("Number of NNs:"))
        gen_layout.addWidget(self.count_spin)
        gen_layout.addWidget(self.generate_btn)
        gen_layout.addWidget(self.clear_btn)
        gen_group.setLayout(gen_layout)

        # Group 2: Benchmarks
        bench_group = QGroupBox("2. Evaluate Benchmarks")
        bench_layout = QVBoxLayout()
        self.bench_btn = QPushButton("Run Benchmarks on All NNs")
        self.bench_btn.clicked.connect(self.run_benchmarks)
        bench_layout.addWidget(self.bench_btn)
        bench_group.setLayout(bench_layout)

        # Group 3: Bayesian Smart Generation (with derivative mode)
        bayes_group = QGroupBox("3. Bayesian Smart Generation (Derivative-Aware)")
        bayes_form = QFormLayout()
        self.smart_count_spin = QSpinBox()
        self.smart_count_spin.setRange(1, 500)
        self.smart_count_spin.setValue(50)
        self.target_combo = QComboBox()
        self.target_combo.addItems(["bench_a", "bench_b", "bench_c", "Any Pass", "All Pass"])

        # NEW: Feature mode for clustering
        self.feature_mode_combo = QComboBox()
        self.feature_mode_combo.addItems([
            "combined (weights + d1-d4)",
            "derivatives_only (d1-d4 behavior)",
            "weights_only (original)"
        ])

        self.smart_gen_btn = QPushButton("Generate Bayesian NNs")
        self.smart_gen_btn.clicked.connect(self.generate_bayesian_nns)
        bayes_form.addRow("Count:", self.smart_count_spin)
        bayes_form.addRow("Target:", self.target_combo)
        bayes_form.addRow("Cluster by:", self.feature_mode_combo)
        bayes_form.addRow(self.smart_gen_btn)
        bayes_group.setLayout(bayes_form)

        # Group 4: Auto-Evolution
        evo_group = QGroupBox("4. Auto-Evolution (Generate + Benchmark + Bayes)")
        evo_layout = QVBoxLayout()
        self.evo_btn = QPushButton("Run 1 Evolution Generation")
        self.evo_btn.setStyleSheet("background-color: #d4edda; font-weight: bold;")
        self.evo_btn.clicked.connect(self.run_evolution_generation)
        evo_layout.addWidget(self.evo_btn)
        evo_group.setLayout(evo_layout)

        # Group 5: Search
        search_group = QGroupBox("5. Query Database")
        search_layout = QVBoxLayout()
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 50)
        self.k_spin.setValue(5)
        self.generate_query_btn = QPushButton("Generate 1 Query NN")
        self.generate_query_btn.clicked.connect(self.generate_query_nn)
        self.search_btn = QPushButton("Find Similar NNs")
        self.search_btn.clicked.connect(self.search_similar)
        self.search_btn.setEnabled(False)
        search_layout.addWidget(QLabel("K Results:"))
        search_layout.addWidget(self.k_spin)
        search_layout.addWidget(self.generate_query_btn)
        search_layout.addWidget(self.search_btn)
        search_group.setLayout(search_layout)

        # Logs & Results
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.on_result_clicked)

        left_layout.addWidget(gen_group)
        left_layout.addWidget(bench_group)
        left_layout.addWidget(bayes_group)
        left_layout.addWidget(evo_group)
        left_layout.addWidget(search_group)
        left_layout.addWidget(QLabel("Activity Log / Stats:"))
        left_layout.addWidget(self.log_console, 2)
        left_layout.addWidget(QLabel("Similar NNs Found (Click to view):"))
        left_layout.addWidget(self.results_list, 1)

        # Right Panel
        right_panel = QSplitter(Qt.Vertical)

        pca_widget = QWidget()
        pca_layout = QVBoxLayout(pca_widget)
        pca_layout.setContentsMargins(0, 0, 0, 0)
        viz_controls_layout = QHBoxLayout()
        viz_controls_layout.addWidget(QLabel("Color Vector Space By:"))
        self.viz_combo = QComboBox()
        self.viz_combo.addItems(["pca_cluster", "derivative_cluster",
                                  "bench_a", "bench_b", "bench_c"])
        self.viz_combo.currentIndexChanged.connect(self.update_pca_plot)
        viz_controls_layout.addWidget(self.viz_combo)
        viz_controls_layout.addStretch()
        pca_layout.addLayout(viz_controls_layout)

        self.vector_canvas = VectorSpaceCanvas()
        self.vector_canvas.pointClicked.connect(self.on_pca_point_clicked)
        self.nn_canvas = NNArchitectureCanvas()
        pca_layout.addWidget(self.vector_canvas)

        right_panel.addWidget(pca_widget)
        right_panel.addWidget(self.nn_canvas)
        right_panel.setSizes([450, 400])

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)

        self.update_stats_log()

    # ── Helper: get current feature mode string ───────────────────────
    def _get_feature_mode(self):
        text = self.feature_mode_combo.currentText()
        if text.startswith("combined"):
            return "combined"
        elif text.startswith("derivatives_only"):
            return "derivatives_only"
        else:
            return "weights_only"

    def log(self, message):
        self.log_console.append(message)

    def _invalidate_deriv_cache(self):
        self._deriv_cache_valid = False

    def _get_deriv_vectors(self):
        if not self._deriv_cache_valid:
            self._deriv_cache = {}
            for i, vec in enumerate(self.db_vectors):
                self._deriv_cache[i] = self.vectorizer.compute_derivative_vector(vec)
            self._deriv_cache_valid = True
        return [self._deriv_cache.get(i, []) for i in range(len(self.db_vectors))]

    def update_stats_log(self):
        total = len(self.db_vectors)
        if total == 0:
            self.log_console.setText("Database is empty. Generate NNs to begin.")
            return

        pass_a = sum(1 for m in self.db_metas if m.get('bench_a') == 'pass')
        pass_b = sum(1 for m in self.db_metas if m.get('bench_b') == 'pass')
        pass_c = sum(1 for m in self.db_metas if m.get('bench_c') == 'pass')
        untested = sum(1 for m in self.db_metas if m.get('bench_a') == 'untested')
        bayes_count = sum(1 for m in self.db_metas if "bayes" in m.get('type', '').lower())

        stats_text = f"--- DATABASE STATS ---\n"
        stats_text += f"Total NNs: {total} (Bayes: {bayes_count}, Untested: {untested})\n\n"
        stats_text += f"Benchmark A Pass Rate: {pass_a}/{total} ({(pass_a/total)*100:.1f}%)\n"
        stats_text += f"Benchmark B Pass Rate: {pass_b}/{total} ({(pass_b/total)*100:.1f}%)\n"
        stats_text += f"Benchmark C Pass Rate: {pass_c}/{total} ({(pass_c/total)*100:.1f}%)\n"

        # ── Derivative statistics ─────────────────────────────────────
        if total > 0 and not untested == total:
            deriv_vecs = self._get_deriv_vectors()
            d1_norms = []
            d2_norms = []
            d3_norms = []
            d4_norms = []
            for dv in deriv_vecs:
                if len(dv) >= 224:
                    half = len(dv) // 4
                    d1_norms.append(np.linalg.norm(dv[:half]))
                    d2_norms.append(np.linalg.norm(dv[half:2*half]))
                    d3_norms.append(np.linalg.norm(dv[2*half:3*half]))
                    d4_norms.append(np.linalg.norm(dv[3*half:]))

            if d1_norms:
                stats_text += f"\n--- DERIVATIVE STATS (Behavioral Fingerprint) ---\n"
                stats_text += f"d1 (Slope)  avg L2: {np.mean(d1_norms):.3f} ± {np.std(d1_norms):.3f}\n"
                stats_text += f"d2 (Curv.)   avg L2: {np.mean(d2_norms):.3f} ± {np.std(d2_norms):.3f}\n"
                stats_text += f"d3 (Jerk)    avg L2: {np.mean(d3_norms):.3f} ± {np.std(d3_norms):.3f}\n"
                stats_text += f"d4 (Snap)    avg L2: {np.mean(d4_norms):.3f} ± {np.std(d4_norms):.3f}\n"

        stats_text += f"----------------------\n"
        self.log_console.setText(stats_text)

    # ── Generation Methods ────────────────────────────────────────────

    def generate_and_store(self):
        count = self.count_spin.value()
        vectors, metadatas, ids = [], [], []
        for i in range(count):
            vec = self.vectorizer.generate_random_nn()
            vectors.append(vec)
            self.db_vectors.append(vec)
            meta = {"type": "Random", "bench_a": "untested",
                    "bench_b": "untested", "bench_c": "untested"}
            metadatas.append(meta)
            self.db_metas.append(meta)
            uid = str(uuid.uuid4())
            ids.append(uid)
            self.db_ids.append(uid)

        self.collection.add(embeddings=vectors, metadatas=metadatas, ids=ids)
        self._invalidate_deriv_cache()
        self.log(f"Added {count} Random NNs.")
        self.update_stats_log()
        self.update_pca_plot()

    def clear_database(self):
        if self.db_ids:
            self.collection.delete(ids=self.db_ids)
        self.db_vectors, self.db_ids, self.db_metas = [], [], []
        self.current_query_vector = None
        self.results_list.clear()
        self.nn_canvas.draw_empty()
        self._invalidate_deriv_cache()
        self.update_pca_plot()
        self.update_stats_log()
        self.log("Database cleared.")

    def run_benchmarks(self):
        if not self.db_vectors:
            return
        for i, vec in enumerate(self.db_vectors):
            if self.db_metas[i].get('bench_a') == 'untested':
                results = self.vectorizer.evaluate_benchmarks(vec)
                self.db_metas[i].update(results)
                # Also compute and cache derivative norms in metadata
                summary = self.vectorizer.get_derivative_summary(vec)
                self.db_metas[i]['d1_norm'] = summary['d1']['l2_norm']
                self.db_metas[i]['d2_norm'] = summary['d2']['l2_norm']
                self.db_metas[i]['d3_norm'] = summary['d3']['l2_norm']
                self.db_metas[i]['d4_norm'] = summary['d4']['l2_norm']

        self.collection.update(ids=self.db_ids, metadatas=self.db_metas)
        self.log("Benchmarks + derivative profiles completed.")
        self.update_stats_log()
        self.update_pca_plot()

    def generate_bayesian_nns(self):
        count = self.smart_count_spin.value()
        target = self.target_combo.currentText()
        feature_mode = self._get_feature_mode()

        successful_vecs = []
        for vec, meta in zip(self.db_vectors, self.db_metas):
            if target == "Any Pass":
                if (meta.get('bench_a') == 'pass' or meta.get('bench_b') == 'pass'
                        or meta.get('bench_c') == 'pass'):
                    successful_vecs.append(vec)
            elif target == "All Pass":
                if (meta.get('bench_a') == 'pass' and meta.get('bench_b') == 'pass'
                        and meta.get('bench_c') == 'pass'):
                    successful_vecs.append(vec)
            else:
                if meta.get(target) == 'pass':
                    successful_vecs.append(vec)

        if not successful_vecs:
            self.log(f"Cannot generate: No NNs in DB passed '{target}'.")
            return

        vectors, metadatas, ids = [], [], []
        pass_count = 0

        for i in range(count):
            new_vec = self.vectorizer.generate_bayesian_nn(successful_vecs, feature_mode)
            vectors.append(new_vec)
            self.db_vectors.append(new_vec)

            results = self.vectorizer.evaluate_benchmarks(new_vec)
            if (results.get(target) == 'pass'
                    or (target == "Any Pass" and 'pass' in results.values())
                    or (target == "All Pass" and all(v == 'pass' for v in results.values()))):
                pass_count += 1

            # Compute derivative summary for metadata
            summary = self.vectorizer.get_derivative_summary(new_vec)
            meta = {
                "type": f"Bayes_{target}",
                "d1_norm": summary['d1']['l2_norm'],
                "d2_norm": summary['d2']['l2_norm'],
                "d3_norm": summary['d3']['l2_norm'],
                "d4_norm": summary['d4']['l2_norm'],
                **results
            }
            metadatas.append(meta)
            self.db_metas.append(meta)
            uid = str(uuid.uuid4())
            ids.append(uid)
            self.db_ids.append(uid)

        self.collection.add(embeddings=vectors, metadatas=metadatas, ids=ids)
        self._invalidate_deriv_cache()
        self.log(f"Generated {count} Bayes NNs (mode={feature_mode}). "
                 f"Pass rate for '{target}': {pass_count}/{count} "
                 f"({(pass_count/count)*100:.1f}%)")
        self.update_stats_log()
        self.update_pca_plot()

    def run_evolution_generation(self):
        """Automates: Random → Benchmark → Bayes (derivative-aware)."""
        self.log("Starting Evolution Generation (derivative-aware)...")

        if len(self.db_vectors) < 50:
            self.count_spin.setValue(50)
            self.generate_and_store()

        self.run_benchmarks()

        self.target_combo.setCurrentText("bench_a")
        self.smart_count_spin.setValue(30)
        self.feature_mode_combo.setCurrentIndex(0)  # combined
        self.generate_bayesian_nns()

        self.log("Evolution Generation Complete.\n")

    def generate_query_nn(self):
        self.current_query_vector = self.vectorizer.generate_random_nn()
        self.search_btn.setEnabled(True)
        self.log("Generated Random Query NN.")
        query_meta = self.vectorizer.evaluate_benchmarks(self.current_query_vector)
        self.nn_canvas.draw_nn(self.current_query_vector,
                               title="Query Neural Network",
                               metadata=query_meta,
                               vectorizer=self.vectorizer)
        self.update_pca_plot()

    def search_similar(self):
        if not self.current_query_vector:
            return
        k = min(self.k_spin.value(), self.collection.count())
        if k == 0:
            return

        results = self.collection.query(
            query_embeddings=[self.current_query_vector],
            n_results=k, include=['distances', 'metadatas']
        )
        self.results_list.clear()
        self.current_result_indices = []

        result_ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]

        for i, rid in enumerate(result_ids):
            db_idx = self.db_ids.index(rid)
            self.current_result_indices.append(db_idx)
            d1n = metadatas[i].get('d1_norm', 0)
            d2n = metadatas[i].get('d2_norm', 0)
            item_text = (f"Dist: {distances[i]:.3f} | Type: {metadatas[i].get('type','?')[:8]} "
                         f"| d1={d1n:.2f} d2={d2n:.2f} (Click)")
            self.results_list.addItem(item_text)

        self.log(f"Found {k} similar NNs.")
        self.update_pca_plot()

    def on_result_clicked(self, item):
        idx = self.results_list.row(item)
        db_idx = self.current_result_indices[idx]
        vec = self.db_vectors[db_idx]
        meta = self.db_metas[db_idx]
        self.nn_canvas.draw_nn(vec,
                               title=f"NN (Type: {meta.get('type')}, Idx: {db_idx})",
                               metadata=meta, vectorizer=self.vectorizer)

    def on_pca_point_clicked(self, db_idx):
        if db_idx < len(self.db_vectors):
            vec = self.db_vectors[db_idx]
            meta = self.db_metas[db_idx]
            self.nn_canvas.draw_nn(vec,
                                   title=f"NN (Type: {meta.get('type')}, Idx: {db_idx})",
                                   metadata=meta, vectorizer=self.vectorizer)
            # Log derivative info
            summary = self.vectorizer.get_derivative_summary(vec)
            self.log(f"Selected NN {db_idx}: "
                     f"|d1|={summary['d1']['l2_norm']:.3f} "
                     f"|d2|={summary['d2']['l2_norm']:.3f} "
                     f"|d3|={summary['d3']['l2_norm']:.3f} "
                     f"|d4|={summary['d4']['l2_norm']:.3f}")

    def update_pca_plot(self, result_indices=None):
        mode = self.viz_combo.currentText()
        self.vector_canvas.plot_pca(
            self.db_vectors, self.db_metas,
            query_vector=self.current_query_vector,
            result_indices=result_indices or self.current_result_indices,
            color_mode=mode,
            vectorizer=self.vectorizer
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())