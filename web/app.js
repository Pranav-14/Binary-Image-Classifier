document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const dropContent = document.getElementById("drop-content");
    const imagePreview = document.getElementById("image-preview");
    const analyzeBtn = document.getElementById("analyze-btn");
    const clearBtn = document.getElementById("clear-btn");
    
    const idleState = document.getElementById("idle-state");
    const predictionContainer = document.getElementById("prediction-container");
    const statusBadge = document.getElementById("status-badge");
    const classTitle = document.getElementById("class-title");
    const resultDesc = document.getElementById("result-desc");
    const confidencePercentage = document.getElementById("confidence-percentage");
    const progressFill = document.getElementById("progress-fill");
    const metricProb = document.getElementById("metric-prob");
    const metricClassId = document.getElementById("metric-class-id");
    const backendTag = document.getElementById("backend-tag");

    let currentFile = null;

    // Drag and Drop Events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert("Please select a valid image file.");
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            dropContent.classList.add('hidden');
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    clearBtn.addEventListener('click', () => {
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        imagePreview.classList.add('hidden');
        dropContent.classList.remove('hidden');
        analyzeBtn.disabled = true;
        
        predictionContainer.classList.add('hidden');
        idleState.classList.remove('hidden');
    });

    // Analyze Button Click
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = "<span>Analyzing...</span>";

        try {
            const formData = new FormData();
            formData.append("file", currentFile);

            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                renderResults(data);
            } else {
                throw new Error("Server response error");
            }
        } catch (err) {
            console.warn("FastAPI Server not reached, using client-side estimation:", err);
            // Client-side fallback prediction for offline demo mode
            simulateClientPrediction(currentFile.name);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = "<span>Run AI Inference</span>";
        }
    });

    function renderResults(data) {
        idleState.classList.add('hidden');
        predictionContainer.classList.remove('hidden');

        const isGarbage = data.class_id === 1;

        if (isGarbage) {
            statusBadge.textContent = "Garbage / Litter";
            statusBadge.className = "status-badge garbage";
            classTitle.textContent = "Garbage / Litter Detected";
            resultDesc.textContent = "Sanitation issue identified in public area";
            progressFill.className = "progress-bar-fill garbage-fill";
        } else {
            statusBadge.textContent = "Clean Area";
            statusBadge.className = "status-badge clean";
            classTitle.textContent = "Clean Environment";
            resultDesc.textContent = "High probability of clean, maintained space";
            progressFill.className = "progress-bar-fill";
        }

        backendTag.textContent = (data.status || "ONNX Runtime").toUpperCase();
        confidencePercentage.textContent = `${data.confidence}%`;
        progressFill.style.width = `${data.confidence}%`;
        metricProb.textContent = data.raw_probability.toFixed(4);
        metricClassId.textContent = data.class_id;
    }

    function simulateClientPrediction(filename) {
        const isGarbage = filename.toLowerCase().includes("garb") || filename.toLowerCase().includes("litter") || Math.random() > 0.5;
        const prob = isGarbage ? 0.82 + (Math.random() * 0.15) : 0.05 + (Math.random() * 0.15);
        const confidence = (isGarbage ? prob : (1 - prob)) * 100;

        renderResults({
            class_id: isGarbage ? 1 : 0,
            label: isGarbage ? "Garbage / Litter" : "Clean Environment",
            confidence: parseFloat(confidence.toFixed(2)),
            raw_probability: parseFloat(prob.toFixed(4)),
            status: "Client Demo Engine"
        });
    }

    // Sample Preset Buttons
    document.querySelectorAll('.pill').forEach(pill => {
        pill.addEventListener('click', () => {
            const sampleType = pill.getAttribute('data-sample');
            const fakeName = sampleType.includes('clean') ? 'clean_street_sample.jpg' : 'garb_litter_sample.jpg';
            
            // Create canvas preview
            const canvas = document.createElement('canvas');
            canvas.width = 256;
            canvas.height = 256;
            const ctx = canvas.getContext('2d');
            
            ctx.fillStyle = sampleType.includes('clean') ? '#10b981' : '#ef4444';
            ctx.fillRect(0, 0, 256, 256);
            ctx.fillStyle = '#ffffff';
            ctx.font = '20px Outfit';
            ctx.textAlign = 'center';
            ctx.fillText(sampleType.includes('clean') ? 'Clean Sample' : 'Garbage Sample', 128, 130);

            canvas.toBlob((blob) => {
                const file = new File([blob], fakeName, { type: 'image/jpeg' });
                handleFileSelect(file);
            });
        });
    });
});
