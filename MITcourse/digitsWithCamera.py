import torch
import torchvision.transforms as transforms
from digits import Net
from PIL import Image
import cv2 
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load the trained model
network = Net()
network.load_state_dict(torch.load('results/model.pth'))
# network.eval()

# Transformation for reformatting camera input
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure it's grayscale
    transforms.Resize((28, 28)),  # Resize to MNIST size
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize as MNIST dataset
])


cap = cv2.VideoCapture(0) # 1 for phone camera

while True:
    ret, frame = cap.read()
    if not ret:
        continue 

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img)  # Apply transformations
    img = img.unsqueeze(0)  # Add batch dimension [1, 1, 28, 28]

    # Pass the image through the model
    with torch.no_grad():
        output = network(img)
        pred = output.argmax(dim=1, keepdim=True).item()  # Get predicted label

    # Display the frame with the predicted digit
    cv2.putText(frame, f"Prediction: {pred}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Digit Recognition", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
