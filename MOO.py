import sys
import FDA.FDA
import numpy as np
from utils import gower
from skfuzzy.cluster import cmeans_predict
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Binary
import skfda.preprocessing.dim_reduction.feature_extraction


#############################################################################
# RESPONSE CURVE GENERATION
#############################################################################

def response_curves(model, Xmap, coords, s, xmin, xmax, num, normalizer, diff=0):
    """Get the response curves of the s-th feature for each input sample
    @param model: Pytorch model.
    @param Xmap: Data samples.
    @param coords: x, y coordinates of the field
    @param s: Index of the feature whose responsivity will be assessed.
    @param xmin: Minimum value each feature can take.
    @param xmax: Maximum value each feature can take.
    @param num: Number of possible values the s-th feature can take.
    @param normalizer: Used to revert normalization
    @param diff: Perturbations created by the GA.
    """
    if num > 30:
        num = 30

    x, y = coords
    patches = []
    ct = 0
    for cx in range(x - 2, x + 3):
        for cy in range(y - 2, y + 3):
            if cx - 2 >= 0 and cx + 3 < Xmap.shape[0] and cy - 2 >= 0 and cy + 3 < Xmap.shape[1]:
                patch = Xmap[cx - 2: cx + 3, cy - 2: cy + 3].transpose((2, 0, 1)).copy()
                if np.count_nonzero(patch[-1, :, :] == 0) < 10:
                    patch = patch[None, :] - diff
                    patch, _ = normalizer.normalization(X=patch[None, :].copy(), Y=None)
                    for i, xs in enumerate(np.linspace(start=xmin, stop=xmax, num=num)):
                        patch[0, 0, s, :, :] = xs
                        patches.append(patch.copy())
                    ct += 1
    # Pass through the NN
    yieldPatches = np.array(model.evaluateFold(valxn=np.array(patches)[:, 0, :, :, :], maxs=None, mins=None, batch_size=2000))
    _, yieldPatches = normalizer.reverse_normalization(X=None, Y=yieldPatches)
    results = np.zeros((ct, num))
    ct, ctpatch = 0, 0
    for cx in range(x - 2, x + 3):
        for cy in range(y - 2, y + 3):
            if cx - 2 >= 0 and cx + 3 < Xmap.shape[0] and cy - 2 >= 0 and cy + 3 < Xmap.shape[1]:
                if np.count_nonzero(Xmap[cx - 2: cx + 3, cy - 2: cy + 3, -1] == 0) < 10:
                    # Calculate relative position of point of interest
                    posx = x - cx
                    posy = y - cy
                    for i, xs in enumerate(np.linspace(start=xmin, stop=xmax, num=num)):
                        if yieldPatches[ctpatch, posx, posy] > 5:
                            results[ct, i] = yieldPatches[ctpatch, posx, posy]
                        ctpatch += 1
                    ct += 1
    Y = np.mean(results, axis=0)

    return Y


#############################################################################
# OBJECTIVE FUNCTIONS
#############################################################################

def L1(new_resp: np.ndarray, c_orig: int, fpca: skfda.preprocessing.dim_reduction.feature_extraction.FPCA,
       cmean, stats):
    """Objective function 1: Modify response curve"""
    # Convert curves into FDatagrid
    new_resp = skfda.FDataGrid(data_matrix=new_resp, grid_points=np.linspace(0, 1, len(new_resp)))
    # Transform curves using fPCA
    new_resp = fpca.transform(new_resp)
    # Apply normalization
    for i in range(new_resp.shape[1]):
        new_resp[:, i] = (new_resp[:, i] - stats[0][i]) / (stats[1][i] - stats[0][i]) * 255
    # Use saved clustering model to assign a label to the transformed curve
    new_cl = cmeans_predict(new_resp.T, cntr_trained=cmean, m=2, error=0.005, maxiter=1000,
                                                   seed=42)[0]
    new_c = np.argmax(new_cl, axis=0)
    # Check if the label changed
    if np.max(new_cl, axis=0) > 0.5:
        return - int(new_c != c_orig)
    else:
        return 0


def L2(x_orig, x_counter, types, ranges):
    """Objective function 2: Modify as few features as possible"""
    comp = []
    for i in range(len(types)):
        if x_orig.ndim <= 2:
            if types[i] == 'real':
                comp.append(int(np.abs(x_orig[i] - x_counter[i]) / ranges[i] > 0.005))
            else:
                comp.append(int(x_orig[i] != x_counter[i]))
        else:
            if types[i] == 'real':
                comp.append(int(np.mean(np.abs(x_orig[0][i] - x_counter[0][i])) / ranges[i] > 0.005))
            else:
                comp.append(int(x_orig[0][i] != x_counter[0][i]))

    return np.sum(comp)


def L3(new_resp, Resp_orig):
    """Objective function 3: Counterfactual explanations should have sound feature value combinations"""
    distances = [FDA.FDA.lp_distance(new_resp, resp) for resp in Resp_orig]
    return sorted(distances)[-1]


def L4(x_new, x_old, types, ranges):
    """Objective function 4: Counterfactual explanations should be close to the original feature values"""
    return np.mean(gower(x_new, x_old, types, ranges))


#############################################################################
# MULTI-OBJECTIVE OPTIMIZATION
#############################################################################

class MixedVarsMOO(ElementwiseProblem):

    def __init__(self, Xmap, coords, x_orig, transformer, model, probs, s, types, xmin, xmax, minS, maxS, num, normalizer,
                 resp_orig=None, c_orig=None, cluster_model=None, centroids=None, **kwargs):
        """Initialize Multi-Objective Optimization object.
        @param x_orig: Original input values
        @param transformer: fPCA object used to transform the response curves
        @param model: Pytorch model.
        @param probs: Probability of mutation of each feature
        @param s: Index of the variable whose responsitivity is being analyzed.
        @param types: Array containing the types of input variables.
        @param xmin: Minimum original value each feature can take.
        @param xmax: Maximum original value each feature can take.
        @param minS: Minimum transformed value the s-th feature can take.
        @param maxS: Maximum transformed original value the s-th feature can take.
        @param num: Number of possible values the s-th feature can take.
        @param normalizer: Statistics used to apply z-score normalization before passing the input through the model.
        @param resp_orig: Original response curve.
        @param c_orig: Original cluster
        @param cluster_model: Cluster model use to determine which management zone this sample corresponds to
        @param centroids: Centroid curves that will be used for comparison.
        """
        # Class variables
        self.x_orig = x_orig
        self.Xmap = Xmap
        self.coords = coords
        self.transformer, self.min_pca, self.max_pca = transformer[0], transformer[1], transformer[2]
        self.model = model
        # self.XObs = XObs
        self.probs = probs
        self.types = types
        self.s = s
        self.minS = minS
        self.maxS = maxS
        self.num = num
        self.normalizer = normalizer
        self.centroids = centroids
        self.ranges = xmax - xmin
        self.c_orig = c_orig
        self.cluster_model = cluster_model

        # Calculate original responsivity
        self.resp_orig = resp_orig

        # Declare optimization variables
        variables = dict()
        for k in [i for i in range(len(self.types)) if i != s]:
            if self.types[k] == 'real':
                var = Real(bounds=(xmin[k], xmax[k]))
            elif self.types[k] == 'integer':
                var = Integer(bounds=(int(xmin[k]), int(xmax[k])))
            elif self.types[k] == 'binary':
                var = Binary()
                var.prob = self.probs[k]  # Assign the probability of mutation of the variable
            else:
                sys.exit("Only accepting real, integer and binary values for now.")
            variables[f"x{k:02}"] = var

        super().__init__(vars=variables, n_obj=3)

    def ind_responsivity(self, s, xmin, xmax, num, diff):
        """Evaluate individual responsivity of an input x"""
        # Obtain N-response curve
        Rcurve = response_curves(self.model, self.Xmap, self.coords, s, xmin, xmax, num, self.normalizer, diff)
        # Curve alignment
        return FDA.FDA.function_alignment(Rcurve[None, :])[0]

    def _evaluate(self, x, out, *args, **kwargs):
        # Generate response curve of the counterfactual
        x = np.array([x[f"x{k:02}"] if k != self.s else self.x_orig[0, self.s, 2, 2] for k in range(len(self.types))])
        diff = self.x_orig[0, :, 2, 2] - x
        diff = np.multiply(np.ones(self.x_orig.shape), np.reshape(diff, (1, self.x_orig.shape[1], 1, 1)))
        x = self.x_orig[0, :, :, :] - diff

        resp = self.ind_responsivity(self.s, self.minS, self.maxS, self.num, diff)

        # Calculate first objective
        f1 = L1(resp, self.c_orig, fpca=self.transformer, cmean=self.cluster_model, stats=[self.min_pca, self.max_pca])
        # Calculate second objective
        f2 = L2(x, self.x_orig, self.types, self.ranges)
        # Calculate THIRD objective
        f3 = L4(x, self.x_orig, self.types, self.ranges)
        out["F"] = [f1, f2, f3]
