## Run
```bash
python3 train_all_models.py > README.md
```

---

## Output
```Bash
Starting training of all models...
Using device: cpu


==================================================
Training MNIST MLP Model
==================================================

Epoch 1/10, Loss: 0.3177, Accuracy: 90.17%
Epoch 2/10, Loss: 0.1660, Accuracy: 94.87%
Epoch 3/10, Loss: 0.1343, Accuracy: 95.79%
Epoch 4/10, Loss: 0.1154, Accuracy: 96.44%
Epoch 5/10, Loss: 0.1072, Accuracy: 96.61%
Epoch 6/10, Loss: 0.0993, Accuracy: 96.97%
Epoch 7/10, Loss: 0.0923, Accuracy: 97.16%
Epoch 8/10, Loss: 0.0869, Accuracy: 97.29%
Epoch 9/10, Loss: 0.0835, Accuracy: 97.30%
Epoch 10/10, Loss: 0.0782, Accuracy: 97.54%
Model weights saved to mnist_mlp.pth

==================================================
Training Diabetes MLP Model
==================================================

Epoch 1/10, Loss: 0.2868, Accuracy: 91.85%
Epoch 2/10, Loss: 0.0446, Accuracy: 98.53%
Epoch 3/10, Loss: 0.0311, Accuracy: 99.04%
Epoch 4/10, Loss: 0.0230, Accuracy: 99.28%
Epoch 5/10, Loss: 0.0205, Accuracy: 99.25%
Epoch 6/10, Loss: 0.0163, Accuracy: 99.43%
Epoch 7/10, Loss: 0.0138, Accuracy: 99.58%
Epoch 8/10, Loss: 0.0119, Accuracy: 99.57%
Epoch 9/10, Loss: 0.0107, Accuracy: 99.68%
Epoch 10/10, Loss: 0.0105, Accuracy: 99.65%
Model weights saved to diabetes_mlp.pth

==================================================
Training Fashion MNIST CNN Model
==================================================

Epoch 1/10, Loss: 0.4357, Accuracy: 84.22%
Epoch 2/10, Loss: 0.2817, Accuracy: 89.64%
Epoch 3/10, Loss: 0.2351, Accuracy: 91.34%
Epoch 4/10, Loss: 0.2062, Accuracy: 92.38%
Epoch 5/10, Loss: 0.1833, Accuracy: 93.21%
Epoch 6/10, Loss: 0.1610, Accuracy: 94.02%
Epoch 7/10, Loss: 0.1413, Accuracy: 94.77%
Epoch 8/10, Loss: 0.1238, Accuracy: 95.40%
Epoch 9/10, Loss: 0.1066, Accuracy: 96.07%
Epoch 10/10, Loss: 0.0931, Accuracy: 96.54%
Model weights saved to fashion_mnist_cnn.pth

Training completed for all models!
```