# Counterfactual Analysis of Neural Networks Used to Create Fertilizer Management Zones

## Description

In Precision Agriculture, the utilization of management zones (MZs) that take into account within-field variability facilitates effective fertilizer management. 
This approach enables the optimization of nitrogen (N) rates to maximize crop yield production and enhance agronomic use efficiency.
However, existing works often neglect the consideration of responsivity to fertilizer as a factor influencing MZ determination. 
In response to this gap, we present a MZ clustering method based on fertilizer responsivity.
We build upon the statement that the responsivity of a given site to the fertilizer rate is described by the shape of its corresponding N fertilizer-yield response (N-response) curve.
Thus, we generate N-response curves for all sites within the field using a convolutional neural network (CNN).
The shape of the approximated N-response curves is then characterized using functional principal component analysis.
Subsequently, a counterfactual explanation (CFE) method is applied to discern the impact of various variables on MZ membership. 
The genetic algorithm-based CFE solves a multi-objective optimization problem and aims to identify the minimum combination of features needed to alter a site's cluster assignment.
Results from two yield prediction datasets indicate that the features with the greatest influence on MZ membership are associated with terrain characteristics that either facilitate or impede fertilizer runoff, such as terrain slope or topographic aspect.


Please read our [IJCNN paper](https://arxiv.org/abs/) for more information.


## Usage

This repository contains the scripts needed to replicate the results of our paper:

* `ManagementZone.py`: Contains the ManagementZone class, which contains two main methods: `cluster` and `CFE`:

    * `cluster`: Clusters a field into a pre-defined number of management zones. Its parameters are:
  
          * `num_clusters`: Number of clusters or management zones.
          * `plot`: If True, will plot the N response curves corresponding to each cluster
          * Returns: A labeled 2-D matrix with the same shape as the field

    * `CFE`: Generates Counterfactual Explanations to analyze management zone membership. Its only parameter is:
  
          * `replace`: If True, replaces previously generated response curves and starts from scratch.
        
  The class can be instantiated and executed as follows:

  ```
  manager = ManagementZones(dataset='FieldB', scratch=False)
  manager.cluster(num_clusters=3, plot=False)
  manager.CFE(replace=False)
  ```
  
* `Trainer.py`: Class used for training the NNs using 10x1 cross-validation.

* `MOO.py`: Implements the multi-objective optimization (MOO) framework and defines the objective functions.

* `DataLoader.py`: Load and pre-process the datasets.
        
* `utils.py`: Additional methods used to transform the data and calculate the metrics.   



# Citation
Use this Bibtex to cite this repository

```
@INPROCEEDINGS{Morales:ManagementZone,
AUTHOR="Giorgio Morales and John W. Sheppard",
TITLE="Counterfactual Analysis of Neural Networks Used to Create Fertilizer Management Zones",
BOOKTITLE="2024 International Joint Conference on Neural Networks (IJCNN) (IJCNN 2024)",
ADDRESS="Yokohama, Japan",
DAYS="30",
MONTH=jun,
YEAR=2024,
}
```
