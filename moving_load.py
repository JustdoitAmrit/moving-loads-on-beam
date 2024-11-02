import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define the Beam class as before
class Beam:
    def __init__(self, young, inertia, length, segments):
        self.young = young
        self.inertia = inertia
        self.length = length
        self.dof = 2
        self.segments = segments
        self.node = self.generate_nodes()
        self.bar = self.generate_bars()
        self.point_load = np.zeros_like(self.node)
        self.distributed_load = np.zeros([len(self.bar), 2])
        self.support = np.ones_like(self.node).astype(int)
        self.force = np.zeros([len(self.bar), 2 * self.dof])
        self.displacement = np.zeros([len(self.bar), 2 * self.dof])

    def generate_nodes(self):
        node_positions = np.linspace(0, self.length, self.segments + 1)
        return np.array([[x, 0] for x in node_positions])

    def generate_bars(self):
        return np.array([[i, i + 1] for i in range(self.segments)])

    def analysis(self):
        nn = len(self.node)
        ne = len(self.bar)
        n_dof = self.dof * nn

        d = self.node[self.bar[:, 1], :] - self.node[self.bar[:, 0], :]
        length = np.sqrt((d**2).sum(axis=1))

        matrix = np.zeros([2 * self.dof, 2 * self.dof])
        k = np.zeros([ne, 2 * self.dof, 2 * self.dof])
        ss = np.zeros([n_dof, n_dof])
        for i in range(ne):
            aux = self.dof * self.bar[i, :]
            index = np.r_[aux[0]:aux[0] + self.dof, aux[1]:aux[1] + self.dof]
            l: float = length[i]
            matrix[0] = [12, 6*l, -12, 6*l]
            matrix[1] = [6*l, 4*l**2, -6*l, 2*l**2]
            matrix[2] = [-12, -6*l, 12, -6*l]
            matrix[3] = [6*l, 2*l**2, -6*l, 4*l**2]
            k[i] = self.young * self.inertia * matrix / l**3
            ss[np.ix_(index, index)] += k[i]

        eq_load_ele = np.zeros([len(self.bar), 2 * self.dof])
        for i in range(ne):
            l: float = length[i]
            pi: float = self.distributed_load[i, 0]
            pf: float = self.distributed_load[i, 1]
            eq_load_ele[i, 0] = l * (21 * pi + 9 * pf) / 60
            eq_load_ele[i, 1] = l * (l * (3 * pi + 2 * pf)) / 60
            eq_load_ele[i, 2] = l * (pi * 9 + 21 * pf) / 60
            eq_load_ele[i, 3] = l * (l * (-2 * pi - 3 * pf)) / 60

        for i in range(ne):
            self.point_load[self.bar[i, 0], 0] += eq_load_ele[i, 0]
            self.point_load[self.bar[i, 0], 1] += eq_load_ele[i, 1]
            self.point_load[self.bar[i, 1], 0] += eq_load_ele[i, 2]
            self.point_load[self.bar[i, 1], 1] += eq_load_ele[i, 3]

        free_dof = self.support.flatten().nonzero()[0]
        kff = ss[np.ix_(free_dof, free_dof)]
        p = self.point_load.flatten()
        pf = p[free_dof]
        uf = np.linalg.solve(kff, pf)
        u = self.support.astype(float).flatten()
        u[free_dof] = uf
        u = u.reshape(nn, self.dof)
        u_ele = np.concatenate((u[self.bar[:, 0]], u[self.bar[:, 1]]), axis=1)
        for i in range(ne):
            self.force[i] = np.dot(k[i], u_ele[i]) - eq_load_ele[i]
            self.displacement[i] = u_ele[i]

    def plot(self, scale=None, ax=None):
        ne = len(self.bar)

        ax[0].cla()
        ax[1].cla()
        ax[2].cla()

        # Original and deformed configurations
        for i in range(ne):
            xi, xf = self.node[self.bar[i, 0], 0], self.node[self.bar[i, 1], 0]
            yi, yf = self.node[self.bar[i, 0], 1], self.node[self.bar[i, 1], 1]
            ax[0].plot([xi, xf], [yi, yf], 'b', linewidth=1)

        for i in range(ne):
            dxi, dxf = self.node[self.bar[i, 0], 0], self.node[self.bar[i, 1], 0]
            dyi = self.node[self.bar[i, 0], 1] + self.displacement[i, 0] * scale
            dyf = self.node[self.bar[i, 1], 1] + self.displacement[i, 2] * scale
            ax[0].plot([dxi, dxf], [dyi, dyf], 'r', linewidth=2)

        ax[0].set_title("Beam Deflection")
        ax[1].set_title("BMD")
        ax[2].set_title("SFD")

        plt.draw()

# Streamlit setup
st.title("Beam Analysis Tool")

# User inputs
E = st.number_input("Young's Modulus (E)", min_value=1e3, value=2e12)
I = st.number_input("Moment of Inertia (I)", min_value=1e-6, value=5e-4)
length = st.number_input("Beam Length (m)", min_value=0.1, value=10.0)
segments = st.number_input("Number of Segments", min_value=1, value=12, step=1)

# Adding supports
support_data = []
with st.expander("Add Supports"):
    add_support = st.button("Add Support")
    support_type = st.selectbox("Support Type", ["Fixed", "Pinned"])
    position = st.slider("Position", 0, segments, 0)
    if add_support:
        support_data.append((position, support_type))
    st.write("Current Supports:")
    for pos, sup_type in support_data:
        st.write(f"Position: {pos}, Type: {sup_type}")

# Create and analyze beam on button click
if st.button("Analyze Beam"):
    beam = Beam(E, I, length, segments)
    
    for pos, sup_type in support_data:
        if sup_type == "Fixed":
            beam.support[pos, :] = 0
        elif sup_type == "Pinned":
            beam.support[pos, 0] = 0

    beam.analysis()
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    beam.plot(scale=100, ax=axs)
    st.pyplot(fig)
