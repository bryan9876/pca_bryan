from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn import datasets

# Código de PCA
class PCA:
    def __init__(self, n_componentes):
        self.n_componentes = n_componentes
        self.componentes = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = np.cov(X.T)
        vectorespropios, valorespropios = np.linalg.eig(cov)
        vectorespropios = vectorespropios.T
        idxs = np.argsort(valorespropios)[::-1]
        vectorespropios = vectorespropios[idxs]
        self.componentes = vectorespropios[:self.n_componentes]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.componentes.T)


# Inicialización de Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Cargar dataset Iris
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # PCA con 2 componentes principales
    n_componentes = int(request.form['n_componentes'])
    pca = PCA(n_componentes)
    pca.fit(X)
    X_projected = pca.transform(X)

    # Graficar los resultados
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.colorbar()

    # Guardar gráfico en formato base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return render_template('index.html', plot_url=plot_url)


if __name__ == '__main__':
    app.run(debug=True)
