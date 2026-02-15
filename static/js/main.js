// Main JavaScript for Bone Fracture Detection System

// Global variables
let selectedFile = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    checkAPIStatus();
});

// Setup event listeners
function setupEventListeners() {
    const fileInput = document.getElementById('fileInput');
    const uploadForm = document.getElementById('uploadForm');
    
    // File selection
    fileInput.addEventListener('change', handleFileSelect);
    
    // Form submission
    uploadForm.addEventListener('submit', handleFormSubmit);
    
    // Drag and drop
    setupDragAndDrop();
}

// Handle file selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    
    if (!file) return;
    
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
        showError('Please select a valid image file (PNG, JPG, JPEG, BMP, or TIFF)');
        return;
    }
    
    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }
    
    selectedFile = file;
    displayPreview(file);
    updateFileLabel(file.name);
}

// Display image preview
function displayPreview(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const preview = document.getElementById('imagePreview');
        const previewContainer = document.getElementById('preview-container');
        
        preview.src = e.target.result;
        previewContainer.style.display = 'block';
    };
    
    reader.readAsDataURL(file);
}

// Update file label
function updateFileLabel(filename) {
    const label = document.getElementById('fileLabel');
    label.textContent = filename;
}

// Remove selected image
function removeImage() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('preview-container').style.display = 'none';
    document.getElementById('fileLabel').textContent = 'Choose X-ray image or drag and drop';
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    if (!selectedFile) {
        showError('Please select an image first');
        return;
    }
    
    // Show loading
    showLoading();
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        // Send request
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        hideLoading();
        showError(error.message);
    }
}

// Show loading state
function showLoading() {
    document.getElementById('uploadForm').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
}

// Hide loading state
function hideLoading() {
    document.getElementById('loadingSection').style.display = 'none';
}

// Display results
function displayResults(data) {
    hideLoading();
    
    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Show demo warning if applicable
    if (data.demo_mode) {
        document.getElementById('demoWarning').style.display = 'flex';
    }
    
    // Update prediction badge
    const badge = document.getElementById('predictionBadge');
    const icon = document.getElementById('predictionIcon');
    const text = document.getElementById('predictionText');
    
    if (data.is_fractured) {
        badge.className = 'prediction-badge fractured';
        icon.className = 'fas fa-exclamation-triangle';
        text.textContent = data.prediction;
    } else {
        badge.className = 'prediction-badge normal';
        icon.className = 'fas fa-check-circle';
        text.textContent = data.prediction;
    }
    
    // Update confidence
    document.getElementById('confidenceValue').textContent = data.confidence.toFixed(1) + '%';
    document.getElementById('probability').textContent = (data.probability * 100).toFixed(2);
    
    // Update timestamp
    document.getElementById('timestamp').textContent = data.timestamp;
    
    // Display original image
    document.getElementById('originalImage').src = data.image_url;
    
    // Display Grad-CAM if available
    if (data.gradcam_url) {
        const gradcamContainer = document.getElementById('gradcamContainer');
        const gradcamImage = document.getElementById('gradcamImage');
        
        gradcamImage.src = data.gradcam_url;
        gradcamContainer.style.display = 'block';
    } else {
        document.getElementById('gradcamContainer').style.display = 'none';
    }
}

// Reset analysis
function resetAnalysis() {
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('uploadForm').style.display = 'block';
    removeImage();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Download report
function downloadReport() {
    showInfo('Report download feature coming soon!');
}

// Setup drag and drop
function setupDragAndDrop() {
    const dropZone = document.querySelector('.file-upload-label');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });
    
    dropZone.addEventListener('drop', handleDrop, false);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    e.currentTarget.style.borderColor = '#2563eb';
    e.currentTarget.style.background = '#eff6ff';
}

function unhighlight(e) {
    e.currentTarget.style.borderColor = '#e5e7eb';
    e.currentTarget.style.background = '#f9fafb';
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        const fileInput = document.getElementById('fileInput');
        fileInput.files = files;
        handleFileSelect({ target: fileInput });
    }
}

// Check API status
async function checkAPIStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        console.log('API Status:', data);
        
        if (!data.model_loaded) {
            console.warn('Model not loaded - running in demo mode');
        }
    } catch (error) {
        console.error('Could not check API status:', error);
    }
}

// Show error message
function showError(message) {
    alert('❌ Error: ' + message);
}

// Show info message
function showInfo(message) {
    alert('ℹ️ ' + message);
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Handle image load errors
document.addEventListener('DOMContentLoaded', function() {
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.addEventListener('error', function() {
            console.error('Failed to load image:', this.src);
        });
    });
});

// Add smooth scroll to all anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// Console welcome message
console.log('%c🏥 Bone Fracture Detection System', 'color: #2563eb; font-size: 20px; font-weight: bold;');
console.log('%cBuilt with TensorFlow, OpenCV, and Flask', 'color: #6b7280; font-size: 12px;');
console.log('%cVersion 1.0.0', 'color: #6b7280; font-size: 12px;');
