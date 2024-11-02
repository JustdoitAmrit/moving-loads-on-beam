import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

class Beam:
    def __init__(self, young, inertia, length, segments):
        self.young = young
        self.inertia = inertia
        self.length = length
        self.segments = segments
        self.dof = 2
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

    def plot(self, scale=100):
        fig, axs = plt.subplots(3, figsize=(10, 10))

        # Plotting original and deformed beam shape
        for i in range(len(self.bar)):
            xi, xf = self.node[self.bar[i, 0], 0], self.node[self.bar[i, 1], 0]
            yi, yf = self.node[self.bar[i, 0], 1], self.node[self.bar[i, 1], 1]
            axs[0].plot([xi, xf], [yi, yf], 'b', linewidth=1)

            dyi = self.node[self.bar[i, 0], 1] + self.displacement[i, 0] * scale
            dyf = self.node[self.bar[i, 1], 1] + self.displacement[i, 2] * scale
            axs[0].plot([xi, xf], [dyi, dyf], 'r', linewidth=2)

        axs[0].set_title("Beam Deflection")
        axs[0].grid()

        # Bending moment plot
        for i in range(len(self.bar)):
            mr_xi, mr_xf = self.node[self.bar[i, 0], 0], self.node[self.bar[i, 1], 0]
            mr_yi = -self.force[i, 1]
            mr_yf = self.force[i, 3]
            axs[1].plot([mr_xi, mr_xi, mr_xf, mr_xf], [0, mr_yi, mr_yf, 0], 'r', linewidth=1)
            axs[1].fill([mr_xi, mr_xi, mr_xf, mr_xf], [0, mr_yi, mr_yf, 0], 'c', alpha=0.3)

        axs[1].set_title("Bending Moment Diagram (BMD)")
        axs[1].grid()

        # Shear force plot
        for i in range(len(self.bar)):
            fr_xi, fr_xf = self.node[self.bar[i, 0], 0], self.node[self.bar[i, 1], 0]
            fr_yi = -self.force[i, 0]
            fr_yf = self.force[i, 2]
            axs[2].plot([fr_xi, fr_xi, fr_xf, fr_xf], [0, fr_yi, fr_yf, 0], 'r', linewidth=1)
            axs[2].fill([fr_xi, fr_xi, fr_xf, fr_xf], [0, fr_yi, fr_yf, 0], 'c', alpha=0.3)

        axs[2].set_title("Shear Force Diagram (SFD)")
        axs[2].grid()

        plt.tight_layout()
        return fig

# Streamlit interface
st.title("Beam Analysis App")

# User input for beam properties
E = st.number_input("Enter Young's Modulus (E)", value=2e12)
I = st.number_input("Enter Moment of Inertia (I)", value=5e-4)
length = st.number_input("Enter the length of the beam", value=10.0)
segments = st.number_input("Enter the number of segments", min_value=1, value=12, step=1)

beam = Beam(E, I, length, int(segments))

# Add supports
st.subheader("Add Supports")
support_type = st.selectbox("Select Support Type", ("Fixed", "Pinned"))
support_position = st.slider("Select Support Position", 0, segments, 0)
add_support = st.button("Add Support")

if add_support:
    if support_type == "Fixed":
        beam.support[support_position, :] = 0
    elif support_type == "Pinned":
        beam.support[support_position, 0] = 0
    st.write(f"Added {support_type} support at position {support_position}")

# Calculate and plot
if st.button("Calculate and Plot"):
    beam.analysis()
    fig = beam.plot(scale=100)
    st.pyplot(fig)

