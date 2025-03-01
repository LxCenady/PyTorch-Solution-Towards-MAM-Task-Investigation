import torch

# Define trainable parameter with initial value satisfying x >= 18
x = torch.tensor([20.0], requires_grad=True)  # Initial value between 18~25
y = lambda: 35 - x  # Automatically satisfies perimeter constraint

# Configure optimizer (projected gradient descent)
optimizer = torch.optim.SGD([x], lr=0.1)

# Training loop
max_epochs = 1000
for epoch in range(max_epochs):
    # Calculate negative area (for maximization)
    area = -x * (35 - x)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Backpropagation
    area.backward()
    
    # Update parameters
    optimizer.step()
    
    # Project onto constraint space: x ∈ [18, 25]
    with torch.no_grad():
        x.clamp_(min=18, max=25)
    
    # Training monitor
    if (epoch+1) % 1 == 0:
        print(f"Epoch {epoch+1}: x = {x.item():.2f}, Area = {-area.item():.2f} m²")

# Final results
optimal_x = x.item()
optimal_y = 35 - optimal_x
print(f"\nOptimal solution: Length = {optimal_x:.2f} m, Width = {optimal_y:.2f} m, Max Area = {optimal_x*optimal_y:.2f} m²")