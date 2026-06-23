// 1. We import the free AI tool from Hugging Face
import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers';

const statusText = document.getElementById('status');
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const resultText = document.getElementById('result');

let classifier;

// 2. Load the AI when the page opens
async function loadAI() {
    try {
        // We load a free, general image classification model
        classifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224');
        statusText.innerText = "AI is ready! Upload a picture of a car.";
        imageUpload.disabled = false; // Turn on the upload button
    } catch (error) {
        statusText.innerText = "Oops, something went wrong loading the AI.";
        console.error(error);
    }
}

// Start the loading process immediately
loadAI();

// 3. What happens when you upload an image
imageUpload.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Show the picture on the screen
    const reader = new FileReader();
    reader.onload = async function(e) {
        imagePreview.src = e.target.result;
        imagePreview.style.display = "block";
        resultText.innerText = "Thinking...";

        try {
            // Ask the AI to guess what it is!
            const results = await classifier(e.target.result);
            
            // The AI gives us a list of guesses. We just take the top one (results[0])
            resultText.innerText = `I think this is a: ${results[0].label}`;
        } catch (error) {
            resultText.innerText = "Sorry, I had trouble analyzing that image.";
        }
    }
    reader.readAsDataURL(file); // This converts the image so the AI can read it
});
