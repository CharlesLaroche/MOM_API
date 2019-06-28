# Machine Learning with Median-Of-Mean (MOM) estimators
Statistical/Machine-Learning project under the supervision of Guillaume Lecué

## Content
For the moment, mom_API contains the MOM adaptation of: LASSO, ElasticNet, Matching Pursuit, Cross-validation, Neural networks (on PyTorch).

## Usage
First, download the folder mom_API, for the moment we did not send it to pip.

#### Machine-learning models
For ElasticNet (same for Lasso):
```python
from momAPI.Linear_models.MOM_ElasticNet import MomElasticNet

model = MomElasticNet(rho=1, lamb=0.01, k=15, max_iter=50, tol=10 ** -4)
model.fit(X, Y)  # training
model.predict(X) # prediction
model.score(X, Y) # score (MSE)
```
#### Deep-learning models

```python
import momAPI.nn.indexed_dataset as indexed_dataset
import momAPI.nn.MOM_training as MOM_training

data_train = indexed_dataset(x_train, y_train)
data_val = indexed_dataset(x_val, y_val)

class nn(torch.nn.Module):
    def __init__(self):
        super(nn, self).__init__()
        self.model = Your model
      
    def forward(self, x):
        return self.model(x)
    
nn_ = nn()
optimizer = Your optimizer
loss_ = Your loss
MOM_nn = MOM_training.MomTraining(nn_ , optimizer , loss_ , n_epochs=100 , batch_size=16, n_hist=100)
MOM_nn.fit(data_train, data_val=data_val)  # Training
MOM_nn.model(X)  # Prediction
```
## Contributors

Corentin Jaumain / Tom Guedon / Charles Laroche
