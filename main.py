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

# Machine Learning & Viz imports
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
        
        self.w1_size = input_size * hidden_size
        self.b1_size = hidden_size
        self.w2_size = hidden_size * output_size
        self.b2_size = output_size
        self.vector_size = self.w1_size + self.b1_size + self.w2_size + self.b2_size

    def generate_random_nn(self):
        w1 = np.random.randn(self.input_size, self.hidden_size)
        b1 = np.random.randn(self.hidden_size)
        w2 = np.random.randn(self.input_size, self.output_size)
        b2 = np.random.randn(self.output_size)
        vector = np.concatenate([w1.flatten(), b1, w2.flatten(), b2])
        return vector.tolist()

    def generate_bayesian_nn(self, successful_vectors):
        """
        Multi-Modal Empirical Bayes sampling.
        Clusters successful networks to avoid 'averaging out' distinct lineages.
        """
        if not successful_vectors:
            return self.generate_random_nn()
            
        data = np.array(successful_vectors)
        
        # Multi-modal handling: If we have enough successful NNs, cluster them
        if len(data) >= 4:
            n_clusters = min(2, len(data) // 2)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(data)
            
            # Pick a random cluster to sample from
            chosen_cluster = np.random.randint(0, n_clusters)
            cluster_data = data[labels == chosen_cluster]
        else:
            cluster_data = data
            
        mean_weights = np.mean(cluster_data, axis=0)
        std_weights = np.std(cluster_data, axis=0)
        std_weights[std_weights < 1e-5] = 1e-5  # Prevent zero variance
        
        new_vector = np.random.normal(mean_weights, std_weights)
        return new_vector.tolist()

    def unflatten_nn(self, vector):
        v = np.array(vector)
        idx = 0
        w1 = v[idx : idx + self.w1_size].reshape(self.input_size, self.hidden_size); idx += self.w1_size
        b1 = v[idx : idx + self.b1_size]; idx += self.b1_size
        w2 = v[idx : idx + self.w2_size].reshape(self.input_size, self.output_size); idx += self.w2_size
        b2 = v[idx : idx + self.b2_size]
        return w1, b1, w2, b2

    def forward_pass(self, vector, x):
        w1, b1, w2, b2 = self.unflatten_nn(vector)
        h = np.maximum(0, np.dot(x, w1) + b1)
        o = 1 / (1 + np.exp(-(np.dot(h, w2) + b2)))
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
        
        # Tooltip annotation
        self.tooltip = self.ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                                        bbox=dict(boxstyle="round,pad=0.4", fc="yellow", alpha=0.9),
                                        arrowprops=dict(arrowstyle="->"))
        self.tooltip.set_visible(False)
        self.draw_empty()

    def draw_empty(self):
        self.ax.clear()
        self.ax.set_title("Vector Space (PCA Projection)")
        self.ax.text(0.5, 0.5, "Database Empty", ha='center', va='center', transform=self.ax.transAxes)
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
            
            text = f"Idx: {closest_idx}\nType: {n_type}\nBench: A:{b_a} B:{b_b} C:{b_c}"
            self.tooltip.xy = (event.xdata, event.ydata)
            self.tooltip.set_text(text)
            self.tooltip.set_visible(True)
            self.draw_idle()
        else:
            if self.tooltip.get_visible():
                self.tooltip.set_visible(False)
                self.draw_idle()

    def plot_pca(self, db_vectors, metadatas, query_vector=None, result_indices=None, color_mode="pca_cluster"):
        self.ax.clear()
        self.db_metas = metadatas  # Save for hover tooltip
        
        if not db_vectors:
            self.db_coords = []
            self.draw_empty()
            return
            
        all_data = np.array(db_vectors)
        if query_vector is not None:
            all_data = np.vstack([all_data, np.array(query_vector)])
            
        pca = PCA(n_components=2)
        coords = pca.fit_transform(all_data)
        
        db_coords = coords[:-1] if query_vector is not None else coords
        self.db_coords = db_coords  # Save for click/hover events
        
        # Separate points by type (Random vs Bayes) for different markers
        types = [m.get('type', 'Random') for m in metadatas]
        is_bayes = ["bayes" in t.lower() for t in types]
        
        if color_mode == "pca_cluster":
            n_clusters = min(4, len(db_coords))
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            clusters = kmeans.fit_predict(db_coords)
            
            # Plot Random (Circles)
            self.ax.scatter(db_coords[~np.array(is_bayes), 0], db_coords[~np.array(is_bayes), 1], 
                            c=clusters[~np.array(is_bayes)], cmap='viridis', marker='o', alpha=0.7, label='Random')
            # Plot Bayes (Triangles)
            self.ax.scatter(db_coords[np.array(is_bayes), 0], db_coords[np.array(is_bayes), 1], 
                            c=clusters[np.array(is_bayes)], cmap='viridis', marker='^', alpha=0.9, label='Bayesian', edgecolors='black')
            self.ax.set_title("Vector Space Clusters (o=Random, ^=Bayes)")
            
        elif color_mode in ["bench_a", "bench_b", "bench_c"]:
            colors = ['green' if m.get(color_mode) == 'pass' else 'red' for m in metadatas]
            self.ax.scatter(db_coords[~np.array(is_bayes), 0], db_coords[~np.array(is_bayes), 1], 
                            c=[c for i,c in enumerate(colors) if not is_bayes[i]], marker='o', alpha=0.7, label='Random')
            self.ax.scatter(db_coords[np.array(is_bayes), 0], db_coords[np.array(is_bayes), 1], 
                            c=[c for i,c in enumerate(colors) if is_bayes[i]], marker='^', alpha=0.9, label='Bayesian', edgecolors='black')
            self.ax.set_title(f"{color_mode.capitalize()} Results (Green=Pass, Red=Fail)")

        if result_indices:
            res_coords = db_coords[result_indices]
            self.ax.scatter(res_coords[:, 0], res_coords[:, 1], c='cyan', s=100, marker='s', label='Search Result', edgecolors='k', zorder=4)
            
        if query_vector is not None:
            q_coord = coords[-1]
            self.ax.scatter(q_coord[0], q_coord[1], c='magenta', marker='*', s=200, label='Query NN', edgecolors='k', zorder=5)
            
        self.ax.legend(fontsize=8)
        self.ax.set_xlabel("PC1"); self.ax.set_ylabel("PC2")
        self.draw()


class NNArchitectureCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.setParent(parent)
        self.draw_empty()

    def draw_empty(self):
        self.ax.clear()
        self.ax.set_title("Neural Network Architecture")
        self.ax.axis('off')
        self.ax.text(0.5, 0.5, "Click a point in the Vector Space\nor Generate a Query NN", ha='center', va='center', transform=self.ax.transAxes)
        self.draw()

    def draw_nn(self, vector, title="Neural Network", metadata=None):
        self.ax.clear()
        self.ax.set_title(title, fontsize=10)
        self.ax.axis('off')
        
        w1, b1, w2, b2 = NeuralNetVectorizer().unflatten_nn(vector)
        x_in, x_hid, x_out = 0.1, 0.5, 0.9
        y_in = np.linspace(0.2, 0.8, 4)
        y_hid = np.linspace(0.1, 0.9, 8)
        y_out = np.linspace(0.35, 0.65, 2)
        
        for i in range(4):
            for h in range(8):
                weight = w1[i, h]; color = 'blue' if weight > 0 else 'red'; alpha = min(abs(weight), 1.0)
                self.ax.plot([x_in, x_hid], [y_in[i], y_hid[h]], color=color, alpha=alpha, linewidth=1)
        for h in range(8):
            for o in range(2):
                weight = w2[h, o]; color = 'blue' if weight > 0 else 'red'; alpha = min(abs(weight), 1.0)
                self.ax.plot([x_hid, x_out], [y_hid[h], y_out[o]], color=color, alpha=alpha, linewidth=1)
                
        sizes_hid = [100 + abs(b)*200 for b in b1]; sizes_out = [100 + abs(b)*200 for b in b2]
        self.ax.scatter([x_in]*4, y_in, s=100, c='black', zorder=5)
        self.ax.scatter([x_hid]*8, y_hid, s=sizes_hid, c='black', zorder=5)
        self.ax.scatter([x_out]*2, y_out, s=sizes_out, c='black', zorder=5)
        
        self.ax.text(x_in, 0.95, 'Input', ha='center'); self.ax.text(x_hid, 0.95, 'Hidden', ha='center'); self.ax.text(x_out, 0.95, 'Output', ha='center')
        self.ax.plot([], [], color='blue', label='Positive Weight')
        self.ax.plot([], [], color='red', label='Negative Weight')
        
        if metadata:
            bench_text = f"A: {metadata.get('bench_a', 'N/A')[:4].upper()}\nB: {metadata.get('bench_b', 'N/A')[:4].upper()}\nC: {metadata.get('bench_c', 'N/A')[:4].upper()}"
            self.ax.text(1.05, 0.5, bench_text, transform=self.ax.transAxes, fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black"))
                         
        self.ax.legend(loc='lower left', fontsize=8)
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Vector DB - Evolution & Bayes Edition")
        self.resize(1280, 860)

        self.vectorizer = NeuralNetVectorizer()
        self.current_query_vector = None
        self.db_vectors, self.db_ids, self.db_metas = [], [], []
        self.current_result_indices = []

        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="neural_networks_evo")

        # UI Setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Panel Setup
        left_panel = QWidget(); left_layout = QVBoxLayout(left_panel); left_panel.setFixedWidth(340)
        
        # Group 1: Random Generation
        gen_group = QGroupBox("1. Random Baseline Generation")
        gen_layout = QVBoxLayout()
        self.count_spin = QSpinBox(); self.count_spin.setRange(1, 1000); self.count_spin.setValue(100)
        self.generate_btn = QPushButton("Generate Random NNs"); self.generate_btn.clicked.connect(self.generate_and_store)
        self.clear_btn = QPushButton("Clear Database"); self.clear_btn.setStyleSheet("color: red;"); self.clear_btn.clicked.connect(self.clear_database)
        gen_layout.addWidget(QLabel("Number of NNs:")); gen_layout.addWidget(self.count_spin)
        gen_layout.addWidget(self.generate_btn); gen_layout.addWidget(self.clear_btn)
        gen_group.setLayout(gen_layout)
        
        # Group 2: Benchmarks
        bench_group = QGroupBox("2. Evaluate Benchmarks")
        bench_layout = QVBoxLayout()
        self.bench_btn = QPushButton("Run Benchmarks on All NNs"); self.bench_btn.clicked.connect(self.run_benchmarks)
        bench_layout.addWidget(self.bench_btn); bench_group.setLayout(bench_layout)

        # Group 3: Bayesian Smart Generation
        bayes_group = QGroupBox("3. Bayesian Smart Generation")
        bayes_form = QFormLayout()
        self.smart_count_spin = QSpinBox(); self.smart_count_spin.setRange(1, 500); self.smart_count_spin.setValue(50)
        self.target_combo = QComboBox()
        self.target_combo.addItems(["bench_a", "bench_b", "bench_c", "Any Pass", "All Pass"])
        self.smart_gen_btn = QPushButton("Generate Bayesian NNs"); self.smart_gen_btn.clicked.connect(self.generate_bayesian_nns)
        bayes_form.addRow("Count:", self.smart_count_spin)
        bayes_form.addRow("Target:", self.target_combo)
        bayes_form.addRow(self.smart_gen_btn)
        bayes_group.setLayout(bayes_form)

        # Group 4: Auto-Evolution
        evo_group = QGroupBox("4. Auto-Evolution (Generate + Benchmark + Bayes)")
        evo_layout = QVBoxLayout()
        self.evo_btn = QPushButton("Run 1 Evolution Generation"); self.evo_btn.setStyleSheet("background-color: #d4edda; font-weight: bold;")
        self.evo_btn.clicked.connect(self.run_evolution_generation)
        evo_layout.addWidget(self.evo_btn)
        evo_group.setLayout(evo_layout)

        # Group 5: Search
        search_group = QGroupBox("5. Query Database")
        search_layout = QVBoxLayout()
        self.k_spin = QSpinBox(); self.k_spin.setRange(1, 50); self.k_spin.setValue(5)
        self.generate_query_btn = QPushButton("Generate 1 Query NN"); self.generate_query_btn.clicked.connect(self.generate_query_nn)
        self.search_btn = QPushButton("Find Similar NNs"); self.search_btn.clicked.connect(self.search_similar); self.search_btn.setEnabled(False)
        search_layout.addWidget(QLabel("K Results:")); search_layout.addWidget(self.k_spin)
        search_layout.addWidget(self.generate_query_btn); search_layout.addWidget(self.search_btn)
        search_group.setLayout(search_layout)

        # Logs & Results
        self.log_console = QTextEdit(); self.log_console.setReadOnly(True)
        self.results_list = QListWidget(); self.results_list.itemClicked.connect(self.on_result_clicked)
        
        left_layout.addWidget(gen_group); left_layout.addWidget(bench_group); left_layout.addWidget(bayes_group); left_layout.addWidget(evo_group); left_layout.addWidget(search_group)
        left_layout.addWidget(QLabel("Activity Log / Stats:")); left_layout.addWidget(self.log_console, 2)
        left_layout.addWidget(QLabel("Similar NNs Found (Click to view):")); left_layout.addWidget(self.results_list, 1)

        # Right Panel Setup
        right_panel = QSplitter(Qt.Vertical)
        
        pca_widget = QWidget(); pca_layout = QVBoxLayout(pca_widget); pca_layout.setContentsMargins(0,0,0,0)
        viz_controls_layout = QHBoxLayout()
        viz_controls_layout.addWidget(QLabel("Color Vector Space By:"))
        self.viz_combo = QComboBox()
        self.viz_combo.addItems(["pca_cluster", "bench_a", "bench_b", "bench_c"])
        self.viz_combo.currentIndexChanged.connect(self.update_pca_plot)
        viz_controls_layout.addWidget(self.viz_combo); viz_controls_layout.addStretch()
        pca_layout.addLayout(viz_controls_layout)
        
        self.vector_canvas = VectorSpaceCanvas()
        self.vector_canvas.pointClicked.connect(self.on_pca_point_clicked)
        self.nn_canvas = NNArchitectureCanvas()
        pca_layout.addWidget(self.vector_canvas)
        
        right_panel.addWidget(pca_widget)
        right_panel.addWidget(self.nn_canvas)
        right_panel.setSizes([450, 350])

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)
        
        self.update_stats_log()

    def log(self, message):
        self.log_console.append(message)

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
        stats_text += f"----------------------\n\n"
        
        self.log_console.setText(stats_text)

    def generate_and_store(self):
        count = self.count_spin.value()
        vectors, metadatas, ids = [], [], []
        for i in range(count):
            vec = self.vectorizer.generate_random_nn()
            vectors.append(vec); self.db_vectors.append(vec)
            meta = {"type": "Random", "bench_a": "untested", "bench_b": "untested", "bench_c": "untested"}
            metadatas.append(meta); self.db_metas.append(meta)
            uid = str(uuid.uuid4()); ids.append(uid); self.db_ids.append(uid)
            
        self.collection.add(embeddings=vectors, metadatas=metadatas, ids=ids)
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
        self.update_pca_plot()
        self.update_stats_log()
        self.log("Database cleared.")

    def run_benchmarks(self):
        if not self.db_vectors: return
        for i, vec in enumerate(self.db_vectors):
            if self.db_metas[i].get('bench_a') == 'untested': # Only test untested to save time
                results = self.vectorizer.evaluate_benchmarks(vec)
                self.db_metas[i].update(results)
        self.collection.update(ids=self.db_ids, metadatas=self.db_metas)
        self.log("Benchmarks completed.")
        self.update_stats_log()
        self.update_pca_plot()

    def generate_bayesian_nns(self):
        count = self.smart_count_spin.value()
        target = self.target_combo.currentText()
        
        successful_vecs = []
        for vec, meta in zip(self.db_vectors, self.db_metas):
            if target == "Any Pass":
                if meta.get('bench_a') == 'pass' or meta.get('bench_b') == 'pass' or meta.get('bench_c') == 'pass':
                    successful_vecs.append(vec)
            elif target == "All Pass":
                if meta.get('bench_a') == 'pass' and meta.get('bench_b') == 'pass' and meta.get('bench_c') == 'pass':
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
            new_vec = self.vectorizer.generate_bayesian_nn(successful_vecs)
            vectors.append(new_vec); self.db_vectors.append(new_vec)
            
            results = self.vectorizer.evaluate_benchmarks(new_vec)
            if results.get(target) == 'pass' or (target == "Any Pass" and 'pass' in results.values()) or (target == "All Pass" and all(v == 'pass' for v in results.values())):
                pass_count += 1
                
            meta = {"type": f"Bayes_{target}", **results}
            metadatas.append(meta); self.db_metas.append(meta)
            uid = str(uuid.uuid4()); ids.append(uid); self.db_ids.append(uid)
            
        self.collection.add(embeddings=vectors, metadatas=metadatas, ids=ids)
        self.log(f"Generated {count} Bayes NNs. Pass rate for '{target}': {pass_count}/{count} ({(pass_count/count)*100:.1f}%)")
        self.update_stats_log()
        self.update_pca_plot()

    def run_evolution_generation(self):
        """Automates the full cycle: Random -> Benchmark -> Bayes Generate."""
        self.log("Starting Evolution Generation...")
        
        # 1. Generate 50 random baseline (if DB is empty or small)
        if len(self.db_vectors) < 50:
            self.count_spin.setValue(50)
            self.generate_and_store()
            
        # 2. Benchmark all
        self.run_benchmarks()
        
        # 3. Generate Bayes offspring based on Bench A
        self.target_combo.setCurrentText("bench_a")
        self.smart_count_spin.setValue(30)
        self.generate_bayesian_nns()
        
        self.log("Evolution Generation Complete.\n")

    def generate_query_nn(self):
        self.current_query_vector = self.vectorizer.generate_random_nn()
        self.search_btn.setEnabled(True)
        self.log("Generated Random Query NN.")
        query_meta = self.vectorizer.evaluate_benchmarks(self.current_query_vector)
        self.nn_canvas.draw_nn(self.current_query_vector, title="Query Neural Network", metadata=query_meta)
        self.update_pca_plot()

    def search_similar(self):
        if not self.current_query_vector: return
        k = min(self.k_spin.value(), self.collection.count())
        if k == 0: return
        
        results = self.collection.query(query_embeddings=[self.current_query_vector], n_results=k, include=['distances', 'metadatas'])
        self.results_list.clear()
        self.current_result_indices = []
        
        result_ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        
        for i, rid in enumerate(result_ids):
            db_idx = self.db_ids.index(rid)
            self.current_result_indices.append(db_idx)
            item_text = f"Dist: {distances[i]:.3f} | Type: {metadatas[i].get('type','?')[:5]} (Click to view)"
            self.results_list.addItem(item_text)
            
        self.log(f"Found {k} similar NNs.")
        self.update_pca_plot()

    def on_result_clicked(self, item):
        idx = self.results_list.row(item)
        db_idx = self.current_result_indices[idx]
        vec = self.db_vectors[db_idx]
        meta = self.db_metas[db_idx]
        self.nn_canvas.draw_nn(vec, title=f"NN (Type: {meta.get('type')}, Idx: {db_idx})", metadata=meta)

    def on_pca_point_clicked(self, db_idx):
        if db_idx < len(self.db_vectors):
            vec = self.db_vectors[db_idx]
            meta = self.db_metas[db_idx]
            self.nn_canvas.draw_nn(vec, title=f"NN (Type: {meta.get('type')}, Idx: {db_idx})", metadata=meta)
            self.log(f"Selected NN {db_idx} from vector space.")

    def update_pca_plot(self, result_indices=None):
        mode = self.viz_combo.currentText()
        self.vector_canvas.plot_pca(
            self.db_vectors, self.db_metas, 
            query_vector=self.current_query_vector, 
            result_indices=result_indices or self.current_result_indices, 
            color_mode=mode
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())