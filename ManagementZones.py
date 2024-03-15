import os
import time
import umap
import utils
import torch
import numpy as np
import pandas as pd
from FDA import FDA
from tqdm import trange
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import combinations
from models.NNModel import NNModel
from skfuzzy.cluster import cmeans
import matplotlib.patches as mpatches
from multiprocessing.pool import ThreadPool
from MOO import MixedVarsMOO, response_curves
from pymoo.core.problem import StarmapParallelization
from Datasets.DataNormalization import DataNormalization

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination.robust import RobustTermination
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, MixedVariableDuplicateElimination


import warnings
warnings.filterwarnings("ignore")


class ManagementZones:

    def __init__(self, dataset='FieldA', scratch=False):
        """
        Produce management zones by clustering
        :param dataset: Field name
        :param scratch: If True, generate the response curves from scratch, otherwise, load pre-computed
        """
        elev, self.elev_str = True, ''
        if elev:
            self.elev_str = '_elev'
            variables = ['yld', 'aa_n', 'slope', 'elev', 'tpi', 'aspect_rad', 'prec_cy_g', 'vv_cy_f', 'vh_cy_f']
        else:
            variables = ['yld', 'aa_n', 'slope', 'elev', 'tpi', 'aspect_rad', 'prec_cy_g', 'vv_cy_f', 'vh_cy_f']

        self.dataset = dataset
        # Load dataset before and after normalization and their corresponding center coordinates
        self.X_orig = np.load('Datasets/' + self.dataset + self.elev_str + '_data_orig.npy', allow_pickle=True)
        self.X_map = np.load('Datasets/' + self.dataset + self.elev_str + '_map.npy', allow_pickle=True)

        # Apply normalization before prediction
        self.names = variables[1:]
        schema = pd.read_csv('Datasets/' + self.dataset + self.elev_str + '_schema.csv')
        self.normalizer = DataNormalization(variables=variables, schema=schema)
        self.X, _ = self.normalizer.normalization(X=self.X_orig.copy(), Y=None)
        self.types = ['real'] * self.X.shape[2]

        # Load x-y coordinates of valid cells within the map
        self.coords = np.load('Datasets/' + self.dataset + self.elev_str + '_coords.npy', allow_pickle=True)

        # Load Model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.reset_model()
        folder = "models//" + self.dataset
        filepath = folder + "//" + self.dataset + self.elev_str + "-model"
        self.model.loadModel(filepath)

        # Process data
        self.validCurves = None  # It will store response curves generated within the field's boundary
        self.xmin, self.xmax, self.num, self.xmin_orig, self.xmax_orig = self._process_data(scratch=scratch)

        # Apply fPCA
        self.Rcurves_transformed, self.fpca = self.fPCA()
        self.min_pca, self.max_pca = np.zeros(self.Rcurves_transformed.shape[1]), np.zeros(self.Rcurves_transformed.shape[1])
        # Normalize curves
        for i in range(self.Rcurves_transformed.shape[1]):
            self.min_pca[i], self.max_pca[i] = np.min(self.Rcurves_transformed[:, i]), np.max(self.Rcurves_transformed[:, i])
            self.Rcurves_transformed[:, i] = (self.Rcurves_transformed[:, i] - self.min_pca[i]) / \
                                             (self.max_pca[i] - self.min_pca[i])
        self.Rcurves_transformed *= 255

        self.grouped_patches = None  # It will store data patches organized by cluster label
        self.grouped_curves = None  # It will store response curves organized by cluster label
        self.grouped_curves_orig = None
        self.grouped_coords = None
        self.kmeans = None
        self.termination = RobustTermination(MultiObjectiveSpaceTermination(tol=1e-5), period=10)

    def reset_model(self):
        return NNModel(device=self.device, nfeatures=self.X.shape[2], method='MCDropout',
                       modelType="Hyper3DNetLiteReg", dataset=self.dataset)

    def _process_data(self, scratch):
        """Prepare data for processing and generate response curves (See https://github.com/GiorgioMorales/ResponsivityAnalysis for a
        # reference on how these response curves are being generated)"""
        # Get minimum and maximum values, and count how many possible values each feature has
        xmin, xmax, num = np.zeros((self.X.shape[2])), np.zeros((self.X.shape[2])), np.zeros((self.X.shape[2]))
        xmin_orig, xmax_orig = np.zeros((self.X.shape[2])), np.zeros((self.X.shape[2]))
        valid_regions = self.X_map[self.X_map[:, :, -1] != 0]
        valid_regions_norm = self.normalizer.normalization(X=valid_regions.copy().transpose(1, 0)[None, None, :, :, None],
                                                           Y=None)[0][0, 0, :, :, 0].transpose(1, 0)
        for f in range(self.X.shape[2]):
            xmin[f], xmax[f], num[f] = np.min(valid_regions_norm[:, f]), np.max(valid_regions_norm[:, f]), len(np.unique(valid_regions[:, f]))
            xmin_orig[f], xmax_orig[f] = np.min(valid_regions[:, f]), np.max(valid_regions[:, f])

        # Get response curves for all points in the dataset:
        if scratch:
            print("Generating response curves for the entire training set...")
            start = time.time()
            s = 0
            Rcurves = []
            for c in trange(len(self.coords)):
                curve = response_curves(model=self.model, Xmap=self.X_map, coords=self.coords[c], s=s, normalizer=self.normalizer,
                                          xmin=xmin[s], xmax=xmax[s], num=num[s])
                Rcurves.append(curve)
            end = time.time()
            print("It took " + str(end - start) + " s. to generate all the response curves.")
            self.validCurves = np.array(Rcurves)
            np.save('ResponseCurves/' + self.dataset + self.elev_str + '_ncurves.npy', self.validCurves)
        else:
            self.validCurves = np.load('ResponseCurves/' + self.dataset + self.elev_str + '_ncurves.npy', allow_pickle=True)

        return xmin, xmax, num, xmin_orig, xmax_orig

    def fPCA(self):
        """
        Apply functional Principal Component Analysis to reduce the dimensionality of the N response curves
        """
        # Align response curves
        self.validCurves = FDA.function_alignment(self.validCurves)
        # Apply FPCA
        return FDA.F_PCA(self.validCurves)

    def cluster(self, num_clusters, plot=True):
        """
        Cluster in a pre-defined number of management zones
        :param num_clusters: num_clusters
        :param plot: If True, will plot the N response curves corresponding to each cluster
        :return: A labeled 2-D matrix with the same shape as the field
        """
        # Fit Fuzzy C-Means model
        self.kmeans, u, u0, d, jm, p, fpc = cmeans(self.Rcurves_transformed.T, c=num_clusters, m=2, error=0.005,
                                                   maxiter=1000, seed=42)
        # Get the fuzzy labels
        labels = np.argmax(u, axis=0)
        membership = np.max(u, axis=0)
        # Organize labels as a 2-D raster
        label_map = np.zeros((self.X_map.shape[0], self.X_map.shape[1])) * np.nan
        membership_map = np.zeros((self.X_map.shape[0], self.X_map.shape[1])) * np.nan
        Rcurves_transformed_2D = np.zeros((self.X_map.shape[0], self.X_map.shape[1], 3)) * np.nan
        for ct, (x, y) in enumerate(self.coords):
            label_map[x, y] = labels[ct]
            membership_map[x, y] = membership[ct]
            Rcurves_transformed_2D[x, y, :] = self.Rcurves_transformed[ct]

        plt.figure()
        im = plt.imshow(label_map, cmap='viridis', interpolation='none')
        colors = [im.cmap(im.norm(value)) for value in range(num_clusters)]
        # create a patch (proxy artist) for every color
        patches = [mpatches.Patch(color=colors[i], label="Cluster {l}".format(l=i + 1)) for i in range(num_clusters)]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xticks([])
        plt.yticks([])

        # Plot curves corresponding to each cluster
        if plot:
            if self.dataset == 'FieldA':
                ylim = 40
            else:
                ylim = 30
            fig, axs = plt.subplots(1, num_clusters, figsize=(5 * num_clusters, 5))
            for lab in range(num_clusters):
                cluster_indices = np.where((labels == lab) & (membership > 0.5))[0]
                # Plot using UMAP
                reducer = umap.UMAP(metric='precomputed', n_components=1, n_neighbors=30, n_epochs=1000)
                distance_matrix = FDA.pairwise_distance(self.validCurves[cluster_indices, :])
                new_order = reducer.fit_transform(distance_matrix)
                # Sort curves according to the new order
                indices = np.argsort(new_order[:, 0])
                # Plot within the same subplot
                ninds = cluster_indices[indices[np.linspace(0, len(cluster_indices) - 1, 50).astype(np.int)]]
                color = mpl.cm.cool(np.linspace(0, 1, len(self.validCurves[ninds, :])))
                for c, g in enumerate(self.validCurves[ninds, :]):
                    axs[lab].plot(np.arange(len(g)) * 5, g, c=color[c], linewidth=2)
                axs[lab].set_ylim(0, ylim)
                axs[lab].set_title(f'Cluster {lab + 1}', fontsize=16)
                axs[lab].tick_params(axis='both', which='both', labelsize=12)
                axs[lab].set_yticks(np.round(np.linspace(0, ylim, 5)))
                axs[lab].set_xticks(np.round(np.linspace(0, 150, 4)))
            # Adjust layout and show the plot
            plt.show()

        # Group patches
        self.grouped_patches, self.grouped_curves, self.grouped_coords, self.grouped_curves_orig = \
            self._group_patches(labels, label_map, membership_map)

    def _group_patches(self, labels, label_map, membership_map):
        """Group 2-D patches according to labels"""
        grouped_patches, grouped_curves, grouped_curves_orig, grouped_coords = {}, {}, {}, {}
        for label in np.unique(labels):
            grouped_patches[label], grouped_curves[label], grouped_curves_orig[label], grouped_coords[label] = [], [], [], []

        for coord, patch, rcurve, rcurve_orig in zip(self.coords, self.X_orig, self.Rcurves_transformed, self.validCurves):
            x, y = coord
            if ~np.isnan(label_map[x - 4: x + 5, y - 4: y + 5]).any() and membership_map[x, y] > 0.5:
                grouped_patches[label_map[x, y]].append(patch)
                grouped_curves[label_map[x, y]].append(rcurve)
                grouped_curves_orig[label_map[x, y]].append(rcurve_orig)
                grouped_coords[label_map[x, y]].append((x, y))
        return grouped_patches, grouped_curves, grouped_coords, grouped_curves_orig

    def CFE(self, replace=True):
        """Generate Counterfactual Explanations to analyze management zones membership
        @param replace: If True, replace previously generated results"""
        nsamples = 200
        # Create matrix that will store the impact of each feature to zones membership
        Ids = np.zeros((nsamples, len(self.types) - 1, len(self.grouped_coords)))
        indexes = [fe for fe in np.arange(len(self.types)) if fe != 0]  # List of passive features (!= s)
        # Create list with all possible combination of feature indexes (up to a max. of 4 features)
        indx = np.arange(len(indexes))
        feat_combs = []
        for v in range(2, 5):
            feat_combs += list(combinations(indx, v))
        combs_matrix = np.zeros((len(feat_combs), len(self.grouped_coords)))
        Ids_cv = np.zeros((len(self.types) - 1, len(self.grouped_coords)))  # Save cross-validation results

        # If the folder does not exist, create it
        folder = "Results/" + self.dataset
        if not os.path.exists(folder):
            os.mkdir(folder)
        impact_results_path = [folder + "/Local_Ids_Nitrogen-Cluster_" + str(lab) + '.npy'
                               for lab in range(len(self.grouped_coords))]

        # Analyze one cluster at a time
        for label in range(len(self.grouped_coords)):
            # Select random samples corresponding to the curent label
            samples = np.random.choice(np.arange(0, len(self.grouped_coords[label])), nsamples, replace=False)
            selected_patches = np.array(self.grouped_patches[label])[samples]
            selected_coords = np.array(self.grouped_coords[label])[samples]
            selected_Rs = np.array(self.grouped_curves[label])[samples]
            selected_Rs_orig = np.array(self.grouped_curves_orig[label])[samples]

            print("\n**************************")
            print("Cluster " + str(label))
            print("**************************")
            # If not all results are already saved, generate response curves
            if not os.path.exists(impact_results_path[label]) or replace:
                for i, (x_orig, coords, rcurve, rcurve_orig) in enumerate(zip(selected_patches, selected_coords, selected_Rs, selected_Rs_orig)):
                    n_threads = 20
                    pool = ThreadPool(n_threads)
                    runner = StarmapParallelization(pool.starmap)

                    probs = [None] * len(self.types)
                    # Define multi-objective optimization problem
                    moo = MixedVarsMOO(Xmap=self.X_map, x_orig=x_orig, coords=coords, model=self.model,
                                       transformer=[self.fpca, self.min_pca, self.max_pca],
                                       probs=probs, resp_orig=rcurve, s=0, types=self.types,
                                       xmin=self.xmin_orig, elementwise_runner=runner,
                                       xmax=self.xmax_orig, minS=self.xmin[0], maxS=self.xmax[0], num=self.num[0],
                                       normalizer=self.normalizer, c_orig=label, cluster_model=self.kmeans)

                    # Optimize using NSGA-II
                    algorithm = NSGA2(pop_size=50,
                                      sampling=MixedVariableSampling(),
                                      mating=MixedVariableMating(
                                          eliminate_duplicates=MixedVariableDuplicateElimination()),
                                      eliminate_duplicates=MixedVariableDuplicateElimination(), )
                    res = minimize(moo, algorithm, ('n_gen', 40), seed=1, verbose=False)
                    # res = minimize(moo, algorithm, self.termination, seed=1, verbose=True)  # ('n_gen', 50)

                    # Select best solution
                    sols = res.F
                    sols[:, 0] = np.round(sols[:, 0], 2)
                    bestf1 = np.min(sols[:, 0])
                    f1sols = sols[(sols[:, 0] == bestf1)]  # Select solutions that produced the highest resp. change
                    bestf2 = np.min(f1sols[:, 1])  # The solution that required to change the fewer features
                    best_sol = np.where((sols[:, 0] == bestf1) & (sols[:, 1] == bestf2))[0][0]
                    x_opt = np.array([res.X[best_sol][f"x{k:02}"] if k != 0 else x_orig[0, 0, 2, 2] for k in
                                      range(len(self.types))])
                    diff = x_orig[0, :, 2, 2] - x_opt
                    diff = np.multiply(np.ones(x_orig.shape), np.reshape(diff, (1, x_orig.shape[1], 1, 1)))
                    x_opt = x_orig[0, :, :, :] - diff
                    dist = utils.gower(x_orig, x_opt, types=self.types, ranges=(self.xmax_orig - self.xmin_orig))
                    Ids[i, :, label] = np.delete(dist, 0)  # Remove feature s
                    print("Sample " + str(i) + ": " + str(Ids[i, :, label] > 0.01))

                    #############################
                    # PLOT ORIGINAL AND CFE
                    #############################
                    # W_orig = self.X_map[coords[0] - 4: coords[0] + 5, coords[1] - 4: coords[1] + 5]
                    # W_cfe = self.X_map[coords[0] - 4: coords[0] + 5, coords[1] - 4: coords[1] + 5] - diff[0, :, 2, 2]
                    # resp_cfe = response_curves(self.model, self.X_map, coords, 0, self.xmin[0], self.xmax[0],
                    #                            self.num[0], normalizer=self.normalizer, diff=diff)
                    # resp_cfe = FDA.function_alignment(resp_cfe[None, :])[0]
                    # plt.figure()
                    # plt.plot(np.arange(len(rcurve_orig)) * 5, rcurve_orig, linewidth=6, label='Original')
                    # plt.plot(np.arange(len(rcurve_orig)) * 5, resp_cfe, alpha=0.7, linewidth=6, label='Counterfactual')
                    # plt.xticks(np.round(np.linspace(0, 150, 4)), fontsize=15)
                    # plt.yticks(np.round(np.linspace(0, 40, 5)), fontsize=15)
                    # plt.legend(fontsize=15)
                    # for v in range(W_orig.shape[2]):
                    #     plt.figure()
                    #     img = plt.imshow(W_orig[:, :, v], vmin=self.xmin_orig[v], vmax=self.xmax_orig[v])
                    #     plt.xticks([])
                    #     plt.yticks([])
                    #     cbar = plt.colorbar(img)
                    #     cbar.ax.tick_params(labelsize=70)
                    #     cbar.set_ticks([np.round(self.xmin_orig[v]), np.round(self.xmax_orig[v])])
                    #     plt.savefig('patch_' + str(v) + '.png', dpi=400)
                    #     plt.figure()
                    #     plt.imshow(W_cfe[:, :, v], vmin=self.xmin_orig[v], vmax=self.xmax_orig[v])
                    #     plt.colorbar()
                    #     plt.xticks([])
                    #     plt.yticks([])
                    #     cbar.ax.tick_params(labelsize=70)
                    #     cbar.set_ticks([np.round(self.xmin_orig[v]), np.round(self.xmax_orig[v])])
                    #     plt.savefig('patchCFE_' + str(v) + '.png', dpi=400)

                    # Save results
                    np.save(impact_results_path[label], Ids[:, :, label])
            else:
                Ids[:, :, label] = np.load(impact_results_path[label])

            ###################
            # Analyze results
            ###################
            Ids_bin = Ids[:, :, label] > 0.005  # Highlight the features that were changed for each sample
            plt.figure()
            plt.imshow(Ids_bin, aspect='auto')
            plt.xticks(np.arange(len(indexes)), [self.names[indexes[v]] for v in range(len(indexes))])
            plt.pause(0.05)

            # Create an array of lists that stores the indexes of the features that changed
            Ilist = np.frompyfunc(list, 0, 1)(np.empty((Ids.shape[0],), dtype=object))
            for n in range(Ids.shape[0]):
                Ilist[n] = list(np.where(Ids_bin[n])[0])
            # Calculate the percentage of times that each variable changed
            percentage = [100 * np.sum(Ids_bin[:, v]) / Ids.shape[0] for v in range(Ids.shape[1])]
            # Retrieve the unique combination of changing variables and how many times they appeared
            combs, comb_num = np.unique(Ilist, return_counts=True)
            sorted_comb_num = np.argsort(comb_num)
            sorted_combinations = combs[sorted_comb_num]  # Combinations of features sorted by repetition
            if len(sorted_combinations) >= 5:
                top5combinations = sorted_combinations[-5:][::-1]
            else:
                top5combinations = sorted_combinations[::-1]

            # Update number of multiple feature combinations
            for vi, v in enumerate(feat_combs):
                combs_matrix[vi, label] = 0
                if list(v) in list(combs):  # If the combination appeared, count how many times
                    p = [p for p, vt in enumerate(combs) if vt == list(v)][0]
                    combs_matrix[vi, label] = comb_num[p]

            # Print results
            print("**************************")
            print("Results for label=" + str(label))
            print("**************************")
            for v in range(len(indexes)):
                print('\033[0m' + "The feature " + str(indexes[v]) + " (" + self.names[
                    indexes[v]] + ") was modified " +
                      '\033[1m' + str(np.round(percentage[v], 2)) + " % of the time." + '\033[0m')
            print("\nThe top-5 most repeated combination of passive features was:")
            for v in range(len(top5combinations)):
                print(
                    "Combination " + str(v) + " : " + str(
                        [self.names[indexes[vi]] for vi in top5combinations[v]]))
            Ids_cv[:, label] = percentage

        plt.figure()
        plt.imshow(Ids_cv[:, :].T, aspect='equal', cmap='Reds')
        for label in range(len(self.grouped_coords)):
            for ic in range(Ids_cv.shape[0]):
                value = Ids_cv[ic, label]
                color = 'white' if value > 40 else 'black'

                plt.text(ic, label, f'{value:.2f}', ha='center', va='center', color=color, weight='bold')
            plt.xticks([])
            plt.yticks([])


if __name__ == '__main__':
    manager = ManagementZones(dataset='FieldB', scratch=False)
    manager.cluster(num_clusters=3, plot=False)
    manager.CFE(replace=False)
