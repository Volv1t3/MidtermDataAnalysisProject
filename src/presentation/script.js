// Slide Navigation Logic
const slides = document.querySelectorAll('.slide');
const btnPrev = document.getElementById('btn-prev');
const btnNext = document.getElementById('btn-next');
const progressBar = document.getElementById('progress-bar');

let currentSlide = 0;
const totalSlides = slides.length;

function updateSlides() {
    slides.forEach((slide, index) => {
        if (index === currentSlide) {
            slide.classList.add('active');
        } else {
            slide.classList.remove('active');
        }
    });

    // Update buttons
    btnPrev.disabled = currentSlide === 0;
    btnNext.disabled = currentSlide === totalSlides - 1;

    // Update progress bar
    const progress = ((currentSlide + 1) / totalSlides) * 100;
    progressBar.style.width = `${progress}%`;
    
    // Trigger 3D scene changes based on slide
    triggerSceneChange(currentSlide);
}

btnPrev.addEventListener('click', () => {
    if (currentSlide > 0) {
        currentSlide--;
        updateSlides();
    }
});

btnNext.addEventListener('click', () => {
    if (currentSlide < totalSlides - 1) {
        currentSlide++;
        updateSlides();
    }
});

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === 'Space') {
        if (currentSlide < totalSlides - 1) {
            currentSlide++;
            updateSlides();
        }
    } else if (e.key === 'ArrowLeft') {
        if (currentSlide > 0) {
            currentSlide--;
            updateSlides();
        }
    }
});

// Three.js Background Implementation
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();

// Camera setup
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 30;
camera.position.y = 10;
camera.lookAt(0, 0, 0);

// Renderer setup
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
// Background color handled by CSS, renderer transparent
container.appendChild(renderer.domElement);

// Create Particle System for "Emotional Flow"
const particleCount = 2000;
const geometry = new THREE.BufferGeometry();
const positions = new Float32Array(particleCount * 3);
const colors = new Float32Array(particleCount * 3);

// Yana Palette colors for particles
const colorTeal = new THREE.Color(0x2DD4BF); // Secondary
const colorCoral = new THREE.Color(0xFF8C7A); // Primary

for (let i = 0; i < particleCount; i++) {
    // Distribute particles in a wide area
    const x = (Math.random() - 0.5) * 100;
    const y = (Math.random() - 0.5) * 20;
    const z = (Math.random() - 0.5) * 50;

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    // Mix colors based on position
    const mixedColor = colorTeal.clone().lerp(colorCoral, Math.random());
    colors[i * 3] = mixedColor.r;
    colors[i * 3 + 1] = mixedColor.g;
    colors[i * 3 + 2] = mixedColor.b;
}

geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

// Create a custom shader material for soft, glowing particles
const material = new THREE.PointsMaterial({
    size: 0.3,
    vertexColors: true,
    transparent: true,
    opacity: 0.6,
    blending: THREE.AdditiveBlending
});

const particleSystem = new THREE.Points(geometry, material);
scene.add(particleSystem);

// Add subtle lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(5, 10, 5);
scene.add(directionalLight);

// Target values for smooth transitions
let targetCameraY = 10;
let targetCameraZ = 30;
let targetRotationSpeed = 0.001;
let currentRotationSpeed = 0.001;

// Function to handle slide-specific 3D scene changes
function triggerSceneChange(slideIndex) {
    switch(slideIndex) {
        case 0: // Hero
            targetCameraY = 10;
            targetCameraZ = 30;
            targetRotationSpeed = 0.002;
            break;
        case 1: // Temporal
            targetCameraY = 20;
            targetCameraZ = 20;
            targetRotationSpeed = 0.005;
            break;
        case 2: // Weekly breakdown
            targetCameraY = 5;
            targetCameraZ = 40;
            targetRotationSpeed = 0.001;
            break;
        case 3: // Funnel
            targetCameraY = 15;
            targetCameraZ = 25;
            targetRotationSpeed = 0.008; // Faster, turbulent
            break;
        case 4: // Churn Grid
            targetCameraY = 0;
            targetCameraZ = 35;
            targetRotationSpeed = 0.002;
            break;
        case 5: // Ecosystem
            targetCameraY = -5;
            targetCameraZ = 20;
            targetRotationSpeed = 0.004;
            break;
        case 6: // Recommendations
            targetCameraY = 10;
            targetCameraZ = 30;
            targetRotationSpeed = 0.001;
            break;
    }
}

// Animation Loop
let time = 0;

function animate() {
    requestAnimationFrame(animate);
    
    time += 0.01;

    // Smoothly interpolate camera position
    camera.position.y += (targetCameraY - camera.position.y) * 0.05;
    camera.position.z += (targetCameraZ - camera.position.z) * 0.05;
    camera.lookAt(0, 0, 0);

    // Smoothly interpolate rotation speed
    currentRotationSpeed += (targetRotationSpeed - currentRotationSpeed) * 0.05;
    
    // Rotate entire particle system slowly
    particleSystem.rotation.y += currentRotationSpeed;
    
    // Wave effect on particles
    const positions = particleSystem.geometry.attributes.position.array;
    for (let i = 0; i < particleCount; i++) {
        const i3 = i * 3;
        const x = positions[i3];
        const z = positions[i3 + 2];
        
        // Add a gentle sine wave motion based on x, z and time
        // We only update the Y coordinate to create a sea-like wave
        // Original Y is roughly around 0
        positions[i3 + 1] = Math.sin(x * 0.1 + time) * 2 + Math.cos(z * 0.1 + time) * 2;
    }
    
    particleSystem.geometry.attributes.position.needsUpdate = true;

    renderer.render(scene, camera);
}

// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// Initialize
animate();
updateSlides();
