import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Upload, Play, Pause, RotateCcw, Download, Settings, Activity, Target, Zap, TrendingDown, Award, AlertCircle, Camera, FileText, Brain, Cpu, Eye, EyeOff, Layers, BarChart3, Sparkles } from 'lucide-react';
import * as THREE from 'three';

const NeuralNetworkVisualizer = () => {
  // State Management
  const [weightMatrix, setWeightMatrix] = useState(null);
  const [lossFunction, setLossFunction] = useState('mse');
  const [optimizer, setOptimizer] = useState('sgd');
  const [isTraining, setIsTraining] = useState(false);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [maxEpochs, setMaxEpochs] = useState(100);
  const [learningRate, setLearningRate] = useState(0.01);
  const [momentum, setMomentum] = useState(0.9);
  const [lossHistory, setLossHistory] = useState([]);
  const [weightHistory, setWeightHistory] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [currentLoss, setCurrentLoss] = useState(0);
  const [bestLoss, setBestLoss] = useState(Infinity);
  const [convergenceRate, setConvergenceRate] = useState(0);
  const [gradientNorm, setGradientNorm] = useState(0);
  const [visualizationMode, setVisualizationMode] = useState('surface');
  const [showContours, setShowContours] = useState(true);
  const [showGradientField, setShowGradientField] = useState(false);
  const [animationSpeed, setAnimationSpeed] = useState(150);
  const [batchSize, setBatchSize] = useState(32);
  const [regularization, setRegularization] = useState('none');
  const [regularizationStrength, setRegularizationStrength] = useState(0.01);
  const [showMetrics, setShowMetrics] = useState(true);
  const [autoSave, setAutoSave] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState('idle');
  
  // Optimizer states
  const [velocity, setVelocity] = useState(null);
  const [adamM, setAdamM] = useState(null);
  const [adamV, setAdamV] = useState(null);
  const [adamT, setAdamT] = useState(0);
  
  // Refs
  const canvasRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const meshRef = useRef(null);
  const markerRef = useRef(null);
  const contourLinesRef = useRef([]);
  const gradientArrowsRef = useRef([]);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const animationIdRef = useRef(null);

  // Apply regularization
  const applyRegularization = useCallback((loss, weights) => {
    if (regularization === 'l1') {
      return loss + regularizationStrength * weights.reduce((sum, w) => sum + Math.abs(w), 0);
    } else if (regularization === 'l2') {
      return loss + regularizationStrength * weights.reduce((sum, w) => sum + w * w, 0) / 2;
    } else if (regularization === 'elastic') {
      const l1 = regularizationStrength * weights.reduce((sum, w) => sum + Math.abs(w), 0);
      const l2 = regularizationStrength * weights.reduce((sum, w) => sum + w * w, 0) / 2;
      return loss + 0.5 * (l1 + l2);
    }
    return loss;
  }, [regularization, regularizationStrength]);

  // Enhanced loss functions with regularization
  const lossFunctions = useMemo(() => ({
    mse: (weights, target = 0) => {
      let loss = weights.reduce((sum, w) => sum + Math.pow(w - target, 2), 0) / weights.length;
      return applyRegularization(loss, weights);
    },
    mae: (weights, target = 0) => {
      let loss = weights.reduce((sum, w) => sum + Math.abs(w - target), 0) / weights.length;
      return applyRegularization(loss, weights);
    },
    crossEntropy: (weights) => {
      const softmax = weights.map(w => Math.exp(Math.min(w, 10)));
      const sumExp = softmax.reduce((a, b) => a + b, 0) + 1e-8;
      const probs = softmax.map(s => s / sumExp);
      let loss = -Math.log(Math.max(probs[0], 1e-15));
      return applyRegularization(loss, weights);
    },
    huber: (weights, target = 0, delta = 1) => {
      let loss = weights.reduce((sum, w) => {
        const diff = Math.abs(w - target);
        return sum + (diff <= delta ? 0.5 * diff * diff : delta * (diff - 0.5 * delta));
      }, 0) / weights.length;
      return applyRegularization(loss, weights);
    },
    logCosh: (weights, target = 0) => {
      let loss = weights.reduce((sum, w) => sum + Math.log(Math.cosh(Math.min(w - target, 10))), 0) / weights.length;
      return applyRegularization(loss, weights);
    },
    focal: (weights, gamma = 2) => {
      const softmax = weights.map(w => Math.exp(Math.min(w, 10)));
      const sumExp = softmax.reduce((a, b) => a + b, 0) + 1e-8;
      const probs = softmax.map(s => s / sumExp);
      let loss = -Math.pow(1 - probs[0], gamma) * Math.log(Math.max(probs[0], 1e-15));
      return applyRegularization(loss, weights);
    },
    kldivergence: (weights, target = 0) => {
      const p = weights.map(w => Math.exp(w));
      const q = weights.map(() => Math.exp(target));
      const pSum = p.reduce((a, b) => a + b, 0) + 1e-8;
      const qSum = q.reduce((a, b) => a + b, 0) + 1e-8;
      const pNorm = p.map(v => v / pSum);
      const qNorm = q.map(v => v / qSum);
      let loss = pNorm.reduce((sum, pi, i) => sum + pi * Math.log(Math.max(pi / qNorm[i], 1e-15)), 0);
      return applyRegularization(loss, weights);
    }
  }), [applyRegularization]);

  // Calculate gradient
  const calculateGradient = useCallback((weights) => {
    const epsilon = 0.001;
    const gradient = weights.map((w, i) => {
      const weightsPlus = [...weights];
      const weightsMinus = [...weights];
      weightsPlus[i] = w + epsilon;
      weightsMinus[i] = w - epsilon;
      const lossPlus = lossFunctions[lossFunction](weightsPlus);
      const lossMinus = lossFunctions[lossFunction](weightsMinus);
      return (lossPlus - lossMinus) / (2 * epsilon);
    });
    
    const norm = Math.sqrt(gradient.reduce((sum, g) => sum + g * g, 0));
    setGradientNorm(norm);
    return gradient;
  }, [lossFunction, lossFunctions]);

  // Enhanced optimizers
  const applyOptimizer = useCallback((weights, gradient, epoch) => {
    let newWeights = [...weights];
    
    switch (optimizer) {
      case 'sgd':
        newWeights = weights.map((w, i) => w - learningRate * gradient[i]);
        break;
        
      case 'momentum':
        if (!velocity) {
          setVelocity(new Array(weights.length).fill(0));
          return newWeights;
        }
        const newVelocity = velocity.map((v, i) => momentum * v - learningRate * gradient[i]);
        setVelocity(newVelocity);
        newWeights = weights.map((w, i) => w + newVelocity[i]);
        break;
        
      case 'adam':
        const beta1 = 0.9;
        const beta2 = 0.999;
        const epsilon = 1e-8;
        
        if (!adamM || !adamV) {
          setAdamM(new Array(weights.length).fill(0));
          setAdamV(new Array(weights.length).fill(0));
          setAdamT(1);
          return newWeights;
        }
        
        const t = adamT + 1;
        setAdamT(t);
        
        const newM = adamM.map((m, i) => beta1 * m + (1 - beta1) * gradient[i]);
        const newV = adamV.map((v, i) => beta2 * v + (1 - beta2) * gradient[i] * gradient[i]);
        
        setAdamM(newM);
        setAdamV(newV);
        
        const mHat = newM.map(m => m / (1 - Math.pow(beta1, t)));
        const vHat = newV.map(v => v / (1 - Math.pow(beta2, t)));
        
        newWeights = weights.map((w, i) => w - learningRate * mHat[i] / (Math.sqrt(vHat[i]) + epsilon));
        break;
        
      case 'rmsprop':
        const decay = 0.9;
        if (!adamV) {
          setAdamV(new Array(weights.length).fill(0));
          return newWeights;
        }
        const newV2 = adamV.map((v, i) => decay * v + (1 - decay) * gradient[i] * gradient[i]);
        setAdamV(newV2);
        newWeights = weights.map((w, i) => w - learningRate * gradient[i] / (Math.sqrt(newV2[i]) + 1e-8));
        break;
        
      case 'adagrad':
        if (!adamV) {
          setAdamV(new Array(weights.length).fill(0));
          return newWeights;
        }
        const newV3 = adamV.map((v, i) => v + gradient[i] * gradient[i]);
        setAdamV(newV3);
        newWeights = weights.map((w, i) => w - learningRate * gradient[i] / (Math.sqrt(newV3[i]) + 1e-8));
        break;
        
      default:
        newWeights = weights.map((w, i) => w - learningRate * gradient[i]);
    }
    
    return newWeights;
  }, [optimizer, learningRate, momentum, velocity, adamM, adamV, adamT]);

  // Initialize 3D scene with enhanced features
  useEffect(() => {
    if (!canvasRef.current) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    scene.fog = new THREE.Fog(0x0a0a0a, 10, 50);

    const camera = new THREE.PerspectiveCamera(75, 800 / 500, 0.1, 1000);
    camera.position.set(10, 8, 10);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ 
      canvas: canvasRef.current,
      preserveDrawingBuffer: true,
      antialias: true,
      alpha: true
    });
    renderer.setSize(800, 500);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;

    // Enhanced lighting setup
    const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 15, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.camera.near = 0.1;
    directionalLight.shadow.camera.far = 50;
    directionalLight.shadow.camera.left = -15;
    directionalLight.shadow.camera.right = 15;
    directionalLight.shadow.camera.top = 15;
    directionalLight.shadow.camera.bottom = -15;
    directionalLight.shadow.mapSize.width = 2048;
    directionalLight.shadow.mapSize.height = 2048;
    scene.add(directionalLight);

    const pointLight1 = new THREE.PointLight(0x00ffff, 0.5, 20);
    pointLight1.position.set(-8, 5, -8);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0xff00ff, 0.3, 20);
    pointLight2.position.set(8, 5, 8);
    scene.add(pointLight2);

    // Enhanced grid
    const gridHelper = new THREE.GridHelper(20, 40, 0x444444, 0x222222);
    scene.add(gridHelper);

    // Axes helper
    const axesHelper = new THREE.AxesHelper(8);
    scene.add(axesHelper);

    sceneRef.current = scene;
    rendererRef.current = renderer;
    cameraRef.current = camera;

    // Enhanced mouse controls
    let mouseDown = false;
    let mouseX = 0;
    let mouseY = 0;
    let targetRotationX = 0;
    let targetRotationY = 0;
    let currentRotationX = 0;
    let currentRotationY = 0;

    const handleMouseDown = (event) => {
      mouseDown = true;
      mouseX = event.clientX;
      mouseY = event.clientY;
    };

    const handleMouseUp = () => {
      mouseDown = false;
    };

    const handleMouseMove = (event) => {
      if (!mouseDown) return;

      const deltaX = event.clientX - mouseX;
      const deltaY = event.clientY - mouseY;

      targetRotationY += deltaX * 0.01;
      targetRotationX += deltaY * 0.01;

      mouseX = event.clientX;
      mouseY = event.clientY;
    };

    const handleWheel = (event) => {
      const scale = event.deltaY > 0 ? 1.1 : 0.9;
      camera.position.multiplyScalar(scale);
      camera.position.clampLength(5, 30);
    };

    canvasRef.current.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mouseup', handleMouseUp);
    window.addEventListener('mousemove', handleMouseMove);
    canvasRef.current.addEventListener('wheel', handleWheel);

    // Enhanced animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);

      // Smooth camera rotation
      currentRotationX += (targetRotationX - currentRotationX) * 0.05;
      currentRotationY += (targetRotationY - currentRotationY) * 0.05;

      const radius = camera.position.length();
      camera.position.x = radius * Math.sin(currentRotationY) * Math.cos(currentRotationX);
      camera.position.y = radius * Math.sin(currentRotationX);
      camera.position.z = radius * Math.cos(currentRotationY) * Math.cos(currentRotationX);
      camera.lookAt(0, 0, 0);

      // Animate point lights
      const time = Date.now() * 0.001;
      pointLight1.position.x = Math.sin(time) * 8;
      pointLight1.position.z = Math.cos(time) * 8;
      pointLight2.position.x = Math.sin(time + Math.PI) * 8;
      pointLight2.position.z = Math.cos(time + Math.PI) * 8;

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      canvasRef.current?.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('mousemove', handleMouseMove);
      canvasRef.current?.removeEventListener('wheel', handleWheel);
      renderer.dispose();
    };
  }, []);

  // Generate example weight matrix
  const generateExampleMatrix = () => {
    const size = 5;
    const matrix = Array(size).fill(0).map(() => 
      Array(size).fill(0).map(() => (Math.random() - 0.5) * 2)
    );
    setWeightMatrix(matrix);
    resetTraining();
    generateLossLandscape(matrix);
  };

  // Handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target.result;
        let matrix;
        
        if (file.name.endsWith('.json')) {
          matrix = JSON.parse(content);
        } else {
          const lines = content.split('\n').filter(line => line.trim());
          matrix = lines.map(line => 
            line.split(/[,\s]+/).map(val => parseFloat(val.trim())).filter(val => !isNaN(val))
          ).filter(row => row.length > 0);
        }

        if (Array.isArray(matrix) && matrix.length > 0) {
          setWeightMatrix(matrix);
          resetTraining();
          generateLossLandscape(matrix);
        }
      } catch (error) {
        alert('Error parsing file. Please ensure it\'s a valid JSON or CSV format.');
      }
    };
    reader.readAsText(file);
  };

  // Generate enhanced 3D loss landscape
  const generateLossLandscape = useCallback((matrix) => {
    if (!matrix || !sceneRef.current || !rendererRef.current) return;

    // Clear previous meshes
    const objectsToRemove = [];
    sceneRef.current.traverse((child) => {
      if (child.userData.isLandscape || child.userData.isContour || child.userData.isGradient) {
        objectsToRemove.push(child);
      }
    });
    objectsToRemove.forEach(obj => {
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
      sceneRef.current.remove(obj);
    });

    const flatWeights = matrix.flat();
    
    // Create loss landscape based on visualization mode
    if (visualizationMode === 'surface' || visualizationMode === 'wireframe') {
      const resolution = 60;
      const geometry = new THREE.PlaneGeometry(15, 15, resolution, resolution);
      const positions = geometry.attributes.position;
      const colors = new Float32Array(positions.count * 3);

      let minLoss = Infinity;
      let maxLoss = -Infinity;
      const lossValues = [];

      // Calculate loss values
      for (let i = 0; i < positions.count; i++) {
        const x = positions.getX(i);
        const z = positions.getZ(i);
        
        const sampleWeights = [x * 0.3, z * 0.3, ...flatWeights.slice(2)];
        const loss = lossFunctions[lossFunction](sampleWeights);
        lossValues.push(loss);
        
        minLoss = Math.min(minLoss, loss);
        maxLoss = Math.max(maxLoss, loss);
      }

      // Apply height and colors
      for (let i = 0; i < positions.count; i++) {
        const loss = lossValues[i];
        const normalizedLoss = maxLoss === minLoss ? 0 : (loss - minLoss) / (maxLoss - minLoss);
        
        // Enhanced height mapping with logarithmic scale for better visualization
        const height = Math.log(1 + loss) * 2;
        positions.setY(i, Math.min(height, 8));
        
        // Enhanced color gradient
        const hue = (1 - normalizedLoss) * 240; // Blue to Red
        const color = new THREE.Color(`hsl(${hue}, 100%, 50%)`);
        colors[i * 3] = color.r;
        colors[i * 3 + 1] = color.g;
        colors[i * 3 + 2] = color.b;
      }

      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geometry.attributes.position.needsUpdate = true;
      geometry.computeVertexNormals();

      const material = visualizationMode === 'wireframe' 
        ? new THREE.MeshBasicMaterial({
            wireframe: true,
            vertexColors: true,
            transparent: true,
            opacity: 0.8
          })
        : new THREE.MeshPhongMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide,
            shininess: 100,
            specular: 0x222222
          });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.rotation.x = -Math.PI / 2;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData.isLandscape = true;
      sceneRef.current.add(mesh);
      meshRef.current = mesh;

      // Add contour lines if enabled
      if (showContours) {
        const contourLevels = 10;
        for (let level = 0; level < contourLevels; level++) {
          const contourValue = minLoss + (level / contourLevels) * (maxLoss - minLoss);
          const contourGeometry = new THREE.BufferGeometry();
          const contourPoints = [];

          // Simple contour extraction
          for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
              const idx = i * (resolution + 1) + j;
              if (Math.abs(lossValues[idx] - contourValue) < (maxLoss - minLoss) / (contourLevels * 2)) {
                const x = positions.getX(idx);
                const y = positions.getY(idx);
                const z = positions.getZ(idx);
                contourPoints.push(x, y + 0.05, z);
              }
            }
          }

          if (contourPoints.length > 0) {
            contourGeometry.setAttribute('position', new THREE.Float32BufferAttribute(contourPoints, 3));
            const contourMaterial = new THREE.LineBasicMaterial({
              color: new THREE.Color(`hsl(${(1 - level / contourLevels) * 240}, 100%, 70%)`),
              linewidth: 2,
              transparent: true,
              opacity: 0.6
            });
            const contourLine = new THREE.Line(contourGeometry, contourMaterial);
            contourLine.rotation.x = -Math.PI / 2;
            contourLine.userData.isContour = true;
            sceneRef.current.add(contourLine);
            contourLinesRef.current.push(contourLine);
          }
        }
      }

      // Add gradient field arrows if enabled
      if (showGradientField) {
        const arrowCount = 20;
        const arrowGroup = new THREE.Group();
        
        for (let i = 0; i < arrowCount; i++) {
          for (let j = 0; j < arrowCount; j++) {
            const x = (i / arrowCount - 0.5) * 14;
            const z = (j / arrowCount - 0.5) * 14;
            
            const sampleWeights = [x * 0.3, z * 0.3, ...flatWeights.slice(2)];
            const gradient = calculateGradient(sampleWeights);
            
            if (gradient[0] !== 0 || gradient[1] !== 0) {
              const direction = new THREE.Vector3(-gradient[0], 0, -gradient[1]).normalize();
              const origin = new THREE.Vector3(x, 0.1, z);
              const length = Math.min(Math.sqrt(gradient[0]**2 + gradient[1]**2) * 2, 0.5);
              
              const arrowHelper = new THREE.ArrowHelper(direction, origin, length, 0x00ff00, length * 0.5, length * 0.3);
              arrowGroup.add(arrowHelper);
            }
          }
        }
        
        arrowGroup.userData.isGradient = true;
        sceneRef.current.add(arrowGroup);
        gradientArrowsRef.current = arrowGroup;
      }
    }

    // Add current position marker
    const markerGeometry = new THREE.SphereGeometry(0.2, 32, 32);
    const markerMaterial = new THREE.MeshPhongMaterial({ 
      color: 0xff0000,
      emissive: 0xff0000,
      emissiveIntensity: 0.5,
      shininess: 100
    });
    const marker = new THREE.Mesh(markerGeometry, markerMaterial);
    
    const currentLossValue = lossFunctions[lossFunction](flatWeights);
    const markerHeight = Math.log(1 + currentLossValue) * 2;
    marker.position.set(flatWeights[0] * 0.3 || 0, Math.min(markerHeight, 8) + 0.2, flatWeights[1] * 0.3 || 0);
    marker.castShadow = true;
    marker.userData.isLandscape = true;
    sceneRef.current.add(marker);
    markerRef.current = marker;

    // Add trajectory path if available
    if (weightHistory.length > 1) {
      const pathGeometry = new THREE.BufferGeometry();
      const pathPositions = [];
      const pathColors = [];

      weightHistory.forEach((weights, index) => {
        const loss = lossFunctions[lossFunction](weights);
        const height = Math.log(1 + loss) * 2;
        pathPositions.push(weights[0] * 0.3 || 0, Math.min(height, 8) + 0.1, weights[1] * 0.3 || 0);
        
        // Color gradient along path
        const t = index / (weightHistory.length - 1);
        const color = new THREE.Color(`hsl(${t * 60 + 180}, 100%, 50%)`);
        pathColors.push(color.r, color.g, color.b);
      });

      pathGeometry.setAttribute('position', new THREE.Float32BufferAttribute(pathPositions, 3));
      pathGeometry.setAttribute('color', new THREE.Float32BufferAttribute(pathColors, 3));
      
      const pathMaterial = new THREE.LineBasicMaterial({ 
        vertexColors: true,
        linewidth: 3,
        transparent: true,
        opacity: 0.8
      });
      
      const pathLine = new THREE.Line(pathGeometry, pathMaterial);
      pathLine.userData.isLandscape = true;
      sceneRef.current.add(pathLine);
    }

    // Update metrics
    setCurrentLoss(currentLossValue);
    if (currentLossValue < bestLoss) {
      setBestLoss(currentLossValue);
    }

  }, [lossFunction, lossFunctions, visualizationMode, showContours, showGradientField, weightHistory, bestLoss, calculateGradient]);

  // Enhanced training step
  const trainStep = useCallback(() => {
    if (!weightMatrix) return;

    const flatWeights = weightMatrix.flat();
    const gradient = calculateGradient(flatWeights);
    
    // Apply optimizer
    const newWeights = applyOptimizer(flatWeights, gradient, currentEpoch);
    
    // Reshape to matrix
    const newMatrix = [];
    const rowSize = weightMatrix[0].length;
    for (let i = 0; i < newWeights.length; i += rowSize) {
      newMatrix.push(newWeights.slice(i, i + rowSize));
    }

    const loss = lossFunctions[lossFunction](newWeights);

    // Calculate convergence metrics
    if (lossHistory.length > 10) {
      const recentLosses = lossHistory.slice(-10).map(h => h.loss);
      const avgRecentLoss = recentLosses.reduce((a, b) => a + b, 0) / recentLosses.length;
      const variance = recentLosses.reduce((sum, l) => sum + Math.pow(l - avgRecentLoss, 2), 0) / recentLosses.length;
      const convergence = Math.max(0, 1 - Math.sqrt(variance) / (avgRecentLoss + 1e-8));
      setConvergenceRate(convergence);
      
      // Early stopping check
      if (convergence > 0.99 && loss < 0.01) {
        setIsTraining(false);
        setTrainingStatus('converged');
      }
    }

    setWeightMatrix(newMatrix);
    setWeightHistory(prev => [...prev.slice(-99), newWeights.slice(0, 2)]);
    setLossHistory(prev => [...prev.slice(-199), { epoch: currentEpoch + 1, loss, gradientNorm }]);
    setCurrentEpoch(prev => prev + 1);

    generateLossLandscape(newMatrix);

    // Auto-save checkpoint
    if (autoSave && currentEpoch % 50 === 0) {
      saveCheckpoint();
    }

  }, [weightMatrix, lossFunction, lossFunctions, currentEpoch, lossHistory, generateLossLandscape, calculateGradient, applyOptimizer, gradientNorm, autoSave]);

  // Training loop
  useEffect(() => {
    if (isTraining && currentEpoch < maxEpochs) {
      const timeout = setTimeout(trainStep, animationSpeed);
      return () => clearTimeout(timeout);
    } else if (currentEpoch >= maxEpochs) {
      setIsTraining(false);
      setTrainingStatus('completed');
      if (isRecording) {
        stopRecording();
      }
    }
  }, [isTraining, currentEpoch, maxEpochs, trainStep, animationSpeed, isRecording]);

  // Recording functions
  const startRecording = () => {
    if (!canvasRef.current) return;

    const stream = canvasRef.current.captureStream(30);
    mediaRecorderRef.current = new MediaRecorder(stream, {
      mimeType: 'video/webm;codecs=vp9',
      videoBitsPerSecond: 2500000
    });

    recordedChunksRef.current = [];

    mediaRecorderRef.current.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunksRef.current.push(event.data);
      }
    };

    mediaRecorderRef.current.onstop = () => {
      const blob = new Blob(recordedChunksRef.current, {
        type: 'video/webm'
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `loss_landscape_${Date.now()}.webm`;
      a.click();
      URL.revokeObjectURL(url);
    };

    mediaRecorderRef.current.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  };

  // Control functions
  const handleStartTraining = () => {
    if (isRecording) {
      startRecording();
    }
    setIsTraining(true);
    setTrainingStatus('training');
  };

  const handleStopTraining = () => {
    setIsTraining(false);
    setTrainingStatus('paused');
    if (isRecording) {
      stopRecording();
    }
  };

  const resetTraining = () => {
    setIsTraining(false);
    setCurrentEpoch(0);
    setLossHistory([]);
    setWeightHistory([]);
    setBestLoss(Infinity);
    setConvergenceRate(0);
    setGradientNorm(0);
    setTrainingStatus('idle');
    setVelocity(null);
    setAdamM(null);
    setAdamV(null);
    setAdamT(0);
    if (isRecording) {
      stopRecording();
    }
  };

  const handleReset = () => {
    resetTraining();
    if (weightMatrix) {
      generateLossLandscape(weightMatrix);
    }
  };

  // Export functions
  const exportData = () => {
    const data = {
      weightMatrix,
      lossHistory,
      weightHistory,
      parameters: {
        lossFunction,
        optimizer,
        learningRate,
        momentum,
        regularization,
        regularizationStrength,
        maxEpochs,
        currentEpoch,
        bestLoss,
        convergenceRate
      }
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `training_data_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const saveCheckpoint = () => {
    const checkpoint = {
      epoch: currentEpoch,
      weights: weightMatrix,
      loss: currentLoss,
      timestamp: new Date().toISOString()
    };
    console.log('Checkpoint saved:', checkpoint);
  };

  const takeScreenshot = () => {
    if (!canvasRef.current) return;
    
    canvasRef.current.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `loss_landscape_${Date.now()}.png`;
      a.click();
      URL.revokeObjectURL(url);
    });
  };

  // Formatting functions
  const formatNumber = (num) => {
    if (num === Infinity) return '∞';
    if (num === -Infinity) return '-∞';
    if (isNaN(num)) return 'NaN';
    if (Math.abs(num) < 0.0001) return num.toExponential(2);
    if (Math.abs(num) > 10000) return num.toExponential(2);
    return num.toFixed(4);
  };

  const getStatusColor = () => {
    switch (trainingStatus) {
      case 'training': return 'text-green-500';
      case 'paused': return 'text-yellow-500';
      case 'completed': return 'text-blue-500';
      case 'converged': return 'text-purple-500';
      default: return 'text-gray-500';
    }
  };

  // Render
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-violet-900 text-white p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-5xl font-bold mb-2 bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent">
            Neural Network Loss Landscape Visualizer
          </h1>
          <p className="text-gray-300">Advanced 3D visualization of optimization trajectories</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          {/* Control Panel */}
          <div className="lg:col-span-1 space-y-4">
            {/* Data Input */}
            <div className="bg-black/40 backdrop-blur-lg rounded-xl p-4 border border-purple-500/30">
              <h2 className="text-lg font-semibold mb-3 flex items-center">
                <Upload className="mr-2 text-cyan-400" size={20} />
                Data Input
              </h2>
              <input
                type="file"
                accept=".json,.csv,.txt"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="w-full p-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg cursor-pointer hover:from-blue-700 hover:to-purple-700 transition-all duration-200 flex items-center justify-center text-sm font-medium"
              >
                <Upload size={16} className="mr-2" />
                Upload Matrix
              </label>
              <button
                onClick={generateExampleMatrix}
                className="w-full mt-2 p-3 bg-gradient-to-r from-green-600 to-teal-600 rounded-lg hover:from-green-700 hover:to-teal-700 transition-all duration-200 text-sm font-medium"
              >
                <Sparkles size={16} className="inline mr-2" />
                Generate Example
              </button>
            </div>

            {/* Training Configuration */}
            <div className="bg-black/40 backdrop-blur-lg rounded-xl p-4 border border-purple-500/30">
              <h2 className="text-lg font-semibold mb-3 flex items-center">
                <Settings className="mr-2 text-purple-400" size={20} />
                Configuration
              </h2>
              
              <div className="space-y-3">
                {/* Loss Function */}
                <div>
                  <label className="block text-xs font-medium mb-1 text-gray-300">Loss Function</label>
                  <select
                    value={lossFunction}
                    onChange={(e) => setLossFunction(e.target.value)}
                    className="w-full p-2 bg-gray-800/50 rounded-lg border border-gray-600 text-sm focus:border-purple-400 transition-colors"
                  >
                    <option value="mse">Mean Squared Error</option>
                    <option value="mae">Mean Absolute Error</option>
                    <option value="crossEntropy">Cross Entropy</option>
                    <option value="huber">Huber Loss</option>
                    <option value="logCosh">Log-Cosh</option>
                    <option value="focal">Focal Loss</option>
                    <option value="kldivergence">KL Divergence</option>
                  </select>
                </div>

                {/* Optimizer */}
                <div>
                  <label className="block text-xs font-medium mb-1 text-gray-300">Optimizer</label>
                  <select
                    value={optimizer}
                    onChange={(e) => setOptimizer(e.target.value)}
                    className="w-full p-2 bg-gray-800/50 rounded-lg border border-gray-600 text-sm focus:border-purple-400 transition-colors"
                  >
                    <option value="sgd">SGD</option>
                    <option value="momentum">Momentum</option>
                    <option value="adam">Adam</option>
                    <option value="rmsprop">RMSprop</option>
                    <option value="adagrad">AdaGrad</option>
                  </select>
                </div>

                {/* Regularization */}
                <div>
                  <label className="block text-xs font-medium mb-1 text-gray-300">Regularization</label>
                  <select
                    value={regularization}
                    onChange={(e) => setRegularization(e.target.value)}
                    className="w-full p-2 bg-gray-800/50 rounded-lg border border-gray-600 text-sm focus:border-purple-400 transition-colors"
                  >
                    <option value="none">None</option>
                    <option value="l1">L1 (Lasso)</option>
                    <option value="l2">L2 (Ridge)</option>
                    <option value="elastic">Elastic Net</option>
                  </select>
                </div>

                {/* Learning Rate */}
                <div>
                  <label className="block text-xs font-medium mb-1 text-gray-300">
                    Learning Rate: <span className="text-cyan-400">{learningRate.toFixed(3)}</span>
                  </label>
                  <input
                    type="range"
                    min="0.0001"
                    max="0.1"
                    step="0.0001"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                {/* Momentum (if applicable) */}
                {optimizer === 'momentum' && (
                  <div>
                    <label className="block text-xs font-medium mb-1 text-gray-300">
                      Momentum: <span className="text-cyan-400">{momentum.toFixed(2)}</span>
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="0.99"
                      step="0.01"
                      value={momentum}
                      onChange={(e) => setMomentum(parseFloat(e.target.value))}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                )}

                {/* Max Epochs */}
                <div>
                  <label className="block text-xs font-medium mb-1 text-gray-300">
                    Max Epochs: <span className="text-green-400">{maxEpochs}</span>
                  </label>
                  <input
                    type="range"
                    min="50"
                    max="1000"
                    step="50"
                    value={maxEpochs}
                    onChange={(e) => setMaxEpochs(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                {/* Animation Speed */}
                <div>
                  <label className="block text-xs font-medium mb-1 text-gray-300">
                    Speed: <span className="text-yellow-400">{animationSpeed}ms</span>
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="500"
                    step="10"
                    value={animationSpeed}
                    onChange={(e) => setAnimationSpeed(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                  />
                </div>
              </div>
            </div>

            {/* Visualization Options */}
            <div className="bg-black/40 backdrop-blur-lg rounded-xl p-4 border border-purple-500/30">
              <h2 className="text-lg font-semibold mb-3 flex items-center">
                <Eye className="mr-2 text-green-400" size={20} />
                Visualization
              </h2>
              
              <div className="space-y-2">
                <div>
                  <label className="block text-xs font-medium mb-1 text-gray-300">Mode</label>
                  <select
                    value={visualizationMode}
                    onChange={(e) => setVisualizationMode(e.target.value)}
                    className="w-full p-2 bg-gray-800/50 rounded-lg border border-gray-600 text-sm focus:border-purple-400 transition-colors"
                  >
                    <option value="surface">Surface</option>
                    <option value="wireframe">Wireframe</option>
                  </select>
                </div>

                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showContours}
                    onChange={(e) => setShowContours(e.target.checked)}
                    className="w-4 h-4 text-purple-600 bg-gray-800 border-gray-600 rounded focus:ring-purple-500"
                  />
                  <span className="text-sm">Show Contour Lines</span>
                </label>

                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showGradientField}
                    onChange={(e) => setShowGradientField(e.target.checked)}
                    className="w-4 h-4 text-purple-600 bg-gray-800 border-gray-600 rounded focus:ring-purple-500"
                  />
                  <span className="text-sm">Show Gradient Field</span>
                </label>

                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={autoSave}
                    onChange={(e) => setAutoSave(e.target.checked)}
                    className="w-4 h-4 text-purple-600 bg-gray-800 border-gray-600 rounded focus:ring-purple-500"
                  />
                  <span className="text-sm">Auto-save Checkpoints</span>
                </label>

                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={isRecording}
                    onChange={(e) => setIsRecording(e.target.checked)}
                    className="w-4 h-4 text-purple-600 bg-gray-800 border-gray-600 rounded focus:ring-purple-500"
                  />
                  <span className="text-sm">Record Training</span>
                </label>
              </div>
            </div>

            {/* Control Buttons */}
            <div className="bg-black/40 backdrop-blur-lg rounded-xl p-4 border border-purple-500/30">
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={handleStartTraining}
                  disabled={!weightMatrix || isTraining}
                  className="p-2 bg-gradient-to-r from-green-600 to-emerald-600 rounded-lg hover:from-green-700 hover:to-emerald-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center text-sm font-medium"
                >
                  <Play size={16} className="mr-1" />
                  Start
                </button>
                
                <button
                  onClick={handleStopTraining}
                  disabled={!isTraining}
                  className="p-2 bg-gradient-to-r from-red-600 to-pink-600 rounded-lg hover:from-red-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center text-sm font-medium"
                >
                  <Pause size={16} className="mr-1" />
                  Stop
                </button>
                
                <button
                  onClick={handleReset}
                  className="p-2 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-all duration-200 flex items-center justify-center text-sm font-medium"
                >
                  <RotateCcw size={16} className="mr-1" />
                  Reset
                </button>
                
                <button
                  onClick={exportData}
                  disabled={!weightMatrix}
                  className="p-2 bg-gradient-to-r from-purple-600 to-violet-600 rounded-lg hover:from-purple-700 hover:to-violet-700 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center text-sm font-medium"
                >
                  <Download size={16} className="mr-1" />
                  Export
                </button>
                
                <button
                  onClick={takeScreenshot}
                  className="p-2 bg-gradient-to-r from-cyan-600 to-teal-600 rounded-lg hover:from-cyan-700 hover:to-teal-700 transition-all duration-200 flex items-center justify-center text-sm font-medium col-span-2"
                >
                  <Camera size={16} className="mr-1" />
                  Screenshot
                </button>
              </div>
            </div>
          </div>

          {/* Visualization Panel */}
          <div className="lg:col-span-3 space-y-4">
            {/* 3D Visualization */}
            <div className="bg-black/40 backdrop-blur-lg rounded-xl p-4 border border-purple-500/30">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-xl font-semibold flex items-center">
                  <Layers className="mr-2 text-cyan-400" size={24} />
                  3D Loss Landscape
                </h2>
                {isTraining && (
                  <span className="px-3 py-1 bg-green-600/30 border border-green-500 text-green-400 text-sm rounded-full animate-pulse flex items-center">
                    <Zap size={14} className="mr-1" />
                    Training...
                  </span>
                )}
              </div>
              
              <div className="relative">
                <canvas
                  ref={canvasRef}
                  width={800}
                  height={500}
                  className="w-full rounded-lg bg-gray-900 shadow-2xl"
                  style={{ maxHeight: '500px' }}
                />
                
                {!weightMatrix && (
                  <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 rounded-lg">
                    <div className="text-center">
                      <Brain className="mx-auto mb-4 text-gray-600" size={48} />
                      <p className="text-gray-400">No data loaded</p>
                      <p className="text-sm text-gray-500 mt-2">Upload a weight matrix or generate an example</p>
                    </div>
                  </div>
                )}
              </div>
              
              <p className="text-xs text-gray-500 mt-2 text-center">
                🖱️ Drag to rotate • Scroll to zoom
              </p>
            </div>

            {/* Metrics Panel */}
            {showMetrics && (
              <div className="bg-black/40 backdrop-blur-lg rounded-xl p-4 border border-purple-500/30">
                <h3 className="text-lg font-semibold mb-3 flex items-center">
                  <BarChart3 className="mr-2 text-orange-400" size={20} />
                  Training Metrics
                </h3>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400">Status</div>
                    <div className={`text-sm font-medium ${getStatusColor()}`}>
                      {trainingStatus.toUpperCase()}
                    </div>
                  </div>
                  
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400">Epoch</div>
                    <div className="text-sm font-medium text-cyan-400">
                      {currentEpoch} / {maxEpochs}
                    </div>
                  </div>
                  
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400">Current Loss</div>
                    <div className="text-sm font-medium text-red-400">
                      {formatNumber(currentLoss)}
                    </div>
                  </div>
                  
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400">Best Loss</div>
                    <div className="text-sm font-medium text-green-400">
                      {formatNumber(bestLoss)}
                    </div>
                  </div>
                  
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400">Gradient Norm</div>
                    <div className="text-sm font-medium text-yellow-400">
                      {formatNumber(gradientNorm)}
                    </div>
                  </div>
                  
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400">Convergence</div>
                    <div className="text-sm font-medium text-purple-400">
                      {(convergenceRate * 100).toFixed(1)}%
                    </div>
                  </div>
                  
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400">Optimizer</div>
                    <div className="text-sm font-medium text-blue-400">
                      {optimizer.toUpperCase()}
                    </div>
                  </div>
                  
                  <div className="bg-gray-800/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400">Loss Function</div>
                    <div className="text-sm font-medium text-indigo-400">
                      {lossFunction.toUpperCase()}
                    </div>
                  </div>
                </div>
                
                <div className="mt-3 space-y-2">
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-xs text-gray-400">Training Progress</span>
                      <span className="text-xs text-blue-400">
                        {((currentEpoch / maxEpochs) * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-blue-500 to-cyan-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${(currentEpoch / maxEpochs) * 100}%` }}
                      />
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-xs text-gray-400">Convergence Rate</span>
                      <span className="text-xs text-purple-400">
                        {(convergenceRate * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${convergenceRate * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Loss History Chart */}
            <div className="bg-black/40 backdrop-blur-lg rounded-xl p-4 border border-purple-500/30">
              <h2 className="text-xl font-semibold mb-3 flex items-center">
                <Activity className="mr-2 text-green-400" size={24} />
                Loss History
              </h2>
              
              {lossHistory.length > 0 ? (
                <div className="h-64 relative">
                  <svg
                    width="100%"
                    height="100%"
                    viewBox="0 0 800 256"
                    className="rounded-lg bg-gray-900/50"
                  >
                    <defs>
                      <linearGradient id="lossGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" style={{stopColor:'#06b6d4', stopOpacity:0.8}} />
                        <stop offset="100%" style={{stopColor:'#06b6d4', stopOpacity:0.1}} />
                      </linearGradient>
                    </defs>
                    
                    {/* Grid */}
                    {[0, 0.25, 0.5, 0.75, 1].map((y) => (
                      <line
                        key={y}
                        x1="50"
                        y1={200 - y * 160 + 20}
                        x2="750"
                        y2={200 - y * 160 + 20}
                        stroke="#374151"
                        strokeWidth="1"
                        strokeDasharray="5,5"
                        opacity="0.3"
                      />
                    ))}
                    
                    {lossHistory.length > 1 && (
                      <>
                        {/* Area under curve */}
                        <path
                          d={`M 50,220 ${lossHistory.map((point, index) => {
                            const x = (index / (Math.max(lossHistory.length - 1, 1))) * 700 + 50;
                            const maxLoss = Math.max(...lossHistory.map(p => p.loss));
                            const minLoss = Math.min(...lossHistory.map(p => p.loss));
                            const normalizedLoss = maxLoss === minLoss ? 0.5 : (point.loss - minLoss) / (maxLoss - minLoss);
                            const y = 200 - (normalizedLoss * 160) + 20;
                            return `L ${x},${y}`;
                          }).join(' ')} L 750,220 Z`}
                          fill="url(#lossGradient)"
                        />
                        
                        {/* Main loss curve */}
                        <path
                          d={`M ${lossHistory.map((point, index) => {
                            const x = (index / (Math.max(lossHistory.length - 1, 1))) * 700 + 50;
                            const maxLoss = Math.max(...lossHistory.map(p => p.loss));
                            const minLoss = Math.min(...lossHistory.map(p => p.loss));
                            const normalizedLoss = maxLoss === minLoss ? 0.5 : (point.loss - minLoss) / (maxLoss - minLoss);
                            const y = 200 - (normalizedLoss * 160) + 20;
                            return index === 0 ? `M ${x},${y}` : `L ${x},${y}`;
                          }).join(' ')}`}
                          fill="none"
                          stroke="#06b6d4"
                          strokeWidth="2"
                        />
                      </>
                    )}
                    
                    {/* Axes */}
                    <line x1="50" y1="20" x2="50" y2="220" stroke="#9ca3af" strokeWidth="2" />
                    <line x1="50" y1="220" x2="750" y2="220" stroke="#9ca3af" strokeWidth="2" />
                    
                    {/* Labels */}
                    <text x="400" y="245" textAnchor="middle" fill="#9ca3af" fontSize="12" fontWeight="bold">
                      Epoch
                    </text>
                    <text x="25" y="120" textAnchor="middle" fill="#9ca3af" fontSize="12" fontWeight="bold" transform="rotate(-90 25 120)">
                      Loss
                    </text>
                    
                    {/* Stats */}
                    {lossHistory.length > 0 && (
                      <>
                        <text x="60" y="15" fill="#10b981" fontSize="10" fontWeight="bold">
                          Best: {formatNumber(Math.min(...lossHistory.map(p => p.loss)))}
                        </text>
                        <text x="680" y="15" fill="#f59e0b" fontSize="10" fontWeight="bold">
                          Current: {formatNumber(lossHistory[lossHistory.length - 1]?.loss || 0)}
                        </text>
                      </>
                    )}
                  </svg>
                </div>
              ) : (
                <div className="h-64 flex items-center justify-center text-gray-500 bg-gray-900/50 rounded-lg">
                  <div className="text-center">
                    <TrendingDown className="mx-auto mb-3 text-gray-600" size={40} />
                    <p className="text-sm">No training data yet</p>
                    <p className="text-xs mt-1">Start training to see the loss curve</p>
                  </div>
                </div>
              )}
              
              {/* Loss Statistics */}
              {lossHistory.length > 0 && (
                <div className="mt-4 grid grid-cols-4 gap-2">
                  <div className="bg-gray-800/50 rounded-lg p-2 text-center">
                    <div className="text-xs text-gray-400">Min Loss</div>
                    <div className="text-sm font-bold text-green-400">
                      {formatNumber(Math.min(...lossHistory.map(p => p.loss)))}
                    </div>
                  </div>
                  <div className="bg-gray-800/50 rounded-lg p-2 text-center">
                    <div className="text-xs text-gray-400">Max Loss</div>
                    <div className="text-sm font-bold text-red-400">
                      {formatNumber(Math.max(...lossHistory.map(p => p.loss)))}
                    </div>
                  </div>
                  <div className="bg-gray-800/50 rounded-lg p-2 text-center">
                    <div className="text-xs text-gray-400">Avg Loss</div>
                    <div className="text-sm font-bold text-blue-400">
                      {formatNumber(lossHistory.reduce((sum, p) => sum + p.loss, 0) / lossHistory.length)}
                    </div>
                  </div>
                  <div className="bg-gray-800/50 rounded-lg p-2 text-center">
                    <div className="text-xs text-gray-400">Improvement</div>
                    <div className="text-sm font-bold text-purple-400">
                      {lossHistory.length > 1 
                        ? ((lossHistory[0].loss - lossHistory[lossHistory.length - 1].loss) / lossHistory[0].loss * 100).toFixed(1) + '%'
                        : 'N/A'}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 text-center text-xs text-gray-500">
          <p>Advanced Neural Network Loss Landscape Visualizer</p>
          <p className="mt-1">Explore optimization trajectories in real-time 3D</p>
        </div>
      </div>
    </div>
  );
};

export default NeuralNetworkVisualizer;