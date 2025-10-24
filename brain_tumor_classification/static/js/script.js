// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const resultArea = document.getElementById('resultArea');
const resetBtn = document.getElementById('resetBtn');
const predictionResult = document.getElementById('predictionResult');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceValue = document.getElementById('confidenceValue');
const probabilities = document.getElementById('probabilities');

// File handling
let selectedFile = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    animateOnScroll();
});

// Event Listeners
function initializeEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());

    // Upload button click
    uploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Reset button
    resetBtn.addEventListener('click', resetDemo);

    // Smooth scrolling for navigation links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// File selection handler
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && validateFile(file)) {
        selectedFile = file;
        processFile(file);
    }
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (validateFile(file)) {
            selectedFile = file;
            processFile(file);
        }
    }
}

// File validation
function validateFile(file) {
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (!validTypes.includes(file.type)) {
        showNotification('Please select a valid image file (JPEG, PNG, GIF)', 'error');
        return false;
    }

    if (file.size > maxSize) {
        showNotification('File size must be less than 10MB', 'error');
        return false;
    }

    return true;
}

// Process selected file
function processFile(file) {
    // Show loading state
    uploadArea.classList.add('loading');
    uploadArea.querySelector('.upload-content').innerHTML = `
        <i class="fas fa-spinner fa-spin"></i>
        <h3>Analyzing Image...</h3>
        <p>Please wait while our AI processes your MRI scan</p>
    `;

    // Create FormData and send to backend
    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error analyzing image: ' + error.message, 'error');
        resetDemo();
    })
    .finally(() => {
        uploadArea.classList.remove('loading');
    });
}

// Display prediction results
function displayResults(data) {
    // Update prediction display
    predictionResult.textContent = data.prediction;
    predictionResult.className = `prediction-value ${getPredictionClass(data.prediction)}`;

    // Update confidence
    const confidence = Math.round(data.confidence * 100);
    confidenceBar.style.width = confidence + '%';
    confidenceValue.textContent = confidence + '%';

    // Update probabilities
    probabilities.innerHTML = '';
    Object.entries(data.probabilities).forEach(([label, prob]) => {
        const probPercent = Math.round(prob * 100);
        const probItem = document.createElement('div');
        probItem.className = 'probability-item';
        probItem.innerHTML = `
            <div class="probability-label">${label}</div>
            <div class="probability-value">${probPercent}%</div>
        `;
        probabilities.appendChild(probItem);
    });

    // Show results
    uploadArea.style.display = 'none';
    resultArea.style.display = 'block';

    // Animate confidence bar
    setTimeout(() => {
        confidenceBar.style.transition = 'width 1s ease';
        confidenceBar.style.width = confidence + '%';
    }, 100);
}

// Get CSS class for prediction
function getPredictionClass(prediction) {
    switch (prediction.toLowerCase()) {
        case 'no tumor':
            return 'success';
        case 'glioma':
        case 'meningioma':
        case 'pituitary tumor':
            return 'warning';
        default:
            return '';
    }
}

// Reset demo
function resetDemo() {
    selectedFile = null;
    fileInput.value = '';
    uploadArea.style.display = 'block';
    resultArea.style.display = 'none';

    uploadArea.classList.remove('success', 'error', 'loading');
    uploadArea.querySelector('.upload-content').innerHTML = `
        <i class="fas fa-cloud-upload-alt"></i>
        <h3>Upload MRI Scan</h3>
        <p>Drag and drop your MRI image here or click to browse</p>
        <input type="file" id="fileInput" accept="image/*" style="display: none;">
        <button class="btn btn-primary" id="uploadBtn">Choose File</button>
    `;

    // Reinitialize event listeners
    initializeEventListeners();
}

// Show notification
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotification = document.querySelector('.notification');
    if (existingNotification) {
        existingNotification.remove();
    }

    // Create notification
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <i class="fas fa-${getNotificationIcon(type)}"></i>
        <span>${message}</span>
        <button class="notification-close">
            <i class="fas fa-times"></i>
        </button>
    `;

    // Add to page
    document.body.appendChild(notification);

    // Show notification
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);

    // Auto hide after 5 seconds
    setTimeout(() => {
        hideNotification(notification);
    }, 5000);

    // Close button
    notification.querySelector('.notification-close').addEventListener('click', () => {
        hideNotification(notification);
    });
}

// Get notification icon
function getNotificationIcon(type) {
    switch (type) {
        case 'success':
            return 'check-circle';
        case 'error':
            return 'exclamation-triangle';
        case 'warning':
            return 'exclamation-circle';
        default:
            return 'info-circle';
    }
}

// Hide notification
function hideNotification(notification) {
    notification.classList.remove('show');
    setTimeout(() => {
        notification.remove();
    }, 300);
}

// Animate elements on scroll
function animateOnScroll() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    // Observe elements
    document.querySelectorAll('.feature-card, .author-card, .performance-table').forEach(el => {
        observer.observe(el);
    });
}

// Mobile navigation toggle
function toggleMobileNav() {
    const navMenu = document.querySelector('.nav-menu');
    const navToggle = document.querySelector('.nav-toggle');

    navMenu.classList.toggle('active');
    navToggle.classList.toggle('active');
}

// Add mobile navigation functionality
document.addEventListener('DOMContentLoaded', function() {
    const navToggle = document.querySelector('.nav-toggle');
    if (navToggle) {
        navToggle.addEventListener('click', toggleMobileNav);
    }
});

// Add CSS for notifications
const notificationCSS = `
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background: white;
    padding: 16px 20px;
    border-radius: 8px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    display: flex;
    align-items: center;
    gap: 12px;
    max-width: 400px;
    transform: translateX(100%);
    opacity: 0;
    transition: all 0.3s ease;
    z-index: 10000;
    border-left: 4px solid var(--primary-color);
}

.notification.show {
    transform: translateX(0);
    opacity: 1;
}

.notification.success {
    border-left-color: var(--success-color);
}

.notification.error {
    border-left-color: var(--danger-color);
}

.notification.warning {
    border-left-color: var(--accent-color);
}

.notification-close {
    background: none;
    border: none;
    cursor: pointer;
    color: var(--gray-400);
    font-size: 14px;
    padding: 4px;
    margin-left: auto;
}

.notification-close:hover {
    color: var(--gray-600);
}

.animate-in {
    animation: slideInUp 0.6s ease forwards;
}

@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (max-width: 768px) {
    .nav-menu {
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: white;
        flex-direction: column;
        padding: 20px;
        box-shadow: var(--shadow-lg);
        transform: translateY(-100%);
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
    }

    .nav-menu.active {
        transform: translateY(0);
        opacity: 1;
        visibility: visible;
    }

    .nav-toggle.active span:nth-child(1) {
        transform: rotate(45deg) translate(5px, 5px);
    }

    .nav-toggle.active span:nth-child(2) {
        opacity: 0;
    }

    .nav-toggle.active span:nth-child(3) {
        transform: rotate(-45deg) translate(7px, -6px);
    }

    .notification {
        left: 20px;
        right: 20px;
        max-width: none;
    }
}
`;

// Add CSS to page
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationCSS;
document.head.appendChild(styleSheet);
