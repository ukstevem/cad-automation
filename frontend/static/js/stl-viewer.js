/**
 * Three.js STL viewer component - reusable 3D preview that fills its container
 *
 * Supports:
 *   - Single STL loading (loadSTL) for backward compatibility
 *   - Multi-mesh scenes (loadScene) with per-mesh color/opacity
 *   - Line geometry overlay (addLines) for weld path highlighting
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

export class STLViewer {
    constructor(containerEl) {
        this.container = containerEl;
        this._disposed = false;

        const rect = containerEl.getBoundingClientRect();
        this.width = rect.width || 400;
        this.height = rect.height || 400;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf8fafc);

        // Camera
        this.camera = new THREE.PerspectiveCamera(50, this.width / this.height, 0.1, 10000);
        this.camera.position.set(0, 0, 100);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        containerEl.appendChild(this.renderer.domElement);

        // Controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.1;

        // Lights
        const ambient = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambient);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(1, 1, 1);
        this.scene.add(dirLight);

        const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
        backLight.position.set(-1, -0.5, -1);
        this.scene.add(backLight);

        // Single-mesh state (backward compat)
        this._mesh = null;
        this._edges = null;

        // Multi-mesh state
        this._meshes = [];      // [{mesh, edges, material}]
        this._lines = [];       // [THREE.LineSegments]
        this._sceneCenter = new THREE.Vector3();

        this._animId = null;
        this._loadGen = 0;

        // Picking
        this._raycaster = new THREE.Raycaster();
        this._ndc = new THREE.Vector2();
        this._onMeshClickCb = null;
        this._onMeshContextCb = null;
        this._pointerDownPos = null;

        const canvas = this.renderer.domElement;
        canvas.addEventListener('pointerdown', (e) => {
            this._pointerDownPos = { x: e.clientX, y: e.clientY, button: e.button };
        });
        canvas.addEventListener('pointerup', (e) => {
            const start = this._pointerDownPos;
            this._pointerDownPos = null;
            if (!start) return;
            const dx = e.clientX - start.x;
            const dy = e.clientY - start.y;
            if (dx * dx + dy * dy > 25) return;     // was a drag, not a click
            if (start.button !== e.button) return;  // mismatched button
            const hit = this._pick(e);
            if (!hit) return;
            if (e.button === 0 && this._onMeshClickCb)   this._onMeshClickCb(hit);
            if (e.button === 2 && this._onMeshContextCb) this._onMeshContextCb(hit, e);
        });
        canvas.addEventListener('contextmenu', (e) => {
            // Prevent default only when a right-click handler is installed — otherwise
            // users lose the browser's normal context menu on the 3D viewer.
            if (this._onMeshContextCb) e.preventDefault();
        });

        this._resizeObserver = new ResizeObserver(() => this._onResize());
        this._resizeObserver.observe(containerEl);

        this._animate();
    }

    /**
     * Register a callback fired when a mesh in a multi-mesh scene is clicked.
     * Callback receives the nodeId stored in mesh.userData.nodeId (set by
     * loadScene from item.nodeId).
     */
    setOnMeshClick(cb) {
        this._onMeshClickCb = cb;
    }

    setOnMeshContextMenu(cb) {
        this._onMeshContextCb = cb;
    }

    _pick(event) {
        if (this._meshes.length === 0) return null;
        const rect = this.renderer.domElement.getBoundingClientRect();
        this._ndc.x =  ((event.clientX - rect.left) / rect.width)  * 2 - 1;
        this._ndc.y = -((event.clientY - rect.top)  / rect.height) * 2 + 1;
        this._raycaster.setFromCamera(this._ndc, this.camera);
        const objects = this._meshes.filter(m => m).map(m => m.mesh);
        const hits = this._raycaster.intersectObjects(objects, false);
        if (hits.length === 0) return null;
        return hits[0].object.userData?.nodeId || null;
    }

    // ── Single-mesh loading (backward compat) ────────────────────────

    async loadSTL(url) {
        const gen = ++this._loadGen;
        const loader = new STLLoader();

        return new Promise((resolve, reject) => {
            loader.load(
                url,
                (geometry) => {
                    if (this._disposed || this._loadGen !== gen) {
                        reject(new Error('load cancelled'));
                        return;
                    }
                    try {
                        this._clearSingle();
                        this._clearMulti();

                        const material = new THREE.MeshPhongMaterial({
                            color: 0x4a90d9,
                            specular: 0x222222,
                            shininess: 40,
                            flatShading: false,
                        });

                        geometry.computeVertexNormals();
                        const mesh = new THREE.Mesh(geometry, material);
                        this._mesh = mesh;
                        this.scene.add(mesh);

                        const edgesGeo = new THREE.EdgesGeometry(geometry, 15);
                        const edgesMat = new THREE.LineBasicMaterial({ color: 0x0d1f33 });
                        this._edges = new THREE.LineSegments(edgesGeo, edgesMat);
                        this.scene.add(this._edges);

                        this._fitCamera(geometry);
                        resolve();
                    } catch (err) {
                        reject(err);
                    }
                },
                undefined,
                (err) => reject(err)
            );
        });
    }

    // ── Multi-mesh scene loading ─────────────────────────────────────

    /**
     * Load multiple STL files into the scene.
     *
     * @param {Array<{url: string, color?: number, opacity?: number, label?: string, placement?: number[]}>} items
     *   placement is an optional 4x4 column-major matrix (16 floats) for positioning
     *   the mesh in the parent assembly's coordinate frame.
     * @returns {Promise<void>} resolves when all meshes are loaded
     */
    async loadScene(items) {
        const gen = ++this._loadGen;
        this._clearSingle();
        this._clearMulti();

        const loader = new STLLoader();
        const combinedBox = new THREE.Box3();
        let loadedCount = 0;

        const loadOne = (item, index) => new Promise((resolve, reject) => {
            loader.load(
                item.url,
                (geometry) => {
                    if (this._disposed || this._loadGen !== gen) {
                        reject(new Error('load cancelled'));
                        return;
                    }
                    try {
                        geometry.computeVertexNormals();

                        const color = item.color != null ? item.color : 0xaaaaaa;
                        const opacity = item.opacity != null ? item.opacity : 1.0;

                        const material = new THREE.MeshPhongMaterial({
                            color,
                            specular: 0x222222,
                            shininess: 40,
                            flatShading: false,
                            transparent: opacity < 1.0,
                            opacity,
                        });

                        const mesh = new THREE.Mesh(geometry, material);
                        if (item.nodeId) {
                            mesh.userData.nodeId = item.nodeId;
                        }

                        // Apply instance placement transform if provided
                        if (item.placement && item.placement.length === 16) {
                            const m4 = new THREE.Matrix4();
                            m4.fromArray(item.placement);
                            mesh.applyMatrix4(m4);
                        }

                        this.scene.add(mesh);

                        // Recompute bounding box after transform
                        geometry.computeBoundingBox();
                        const meshBox = geometry.boundingBox.clone();
                        if (item.placement && item.placement.length === 16) {
                            meshBox.applyMatrix4(mesh.matrix);
                        }
                        combinedBox.union(meshBox);

                        const edgesGeo = new THREE.EdgesGeometry(geometry, 15);
                        const edgesMat = new THREE.LineBasicMaterial({
                            color: 0x333333,
                            transparent: opacity < 1.0,
                            opacity: Math.min(opacity + 0.1, 1.0),
                        });
                        const edges = new THREE.LineSegments(edgesGeo, edgesMat);
                        if (item.placement && item.placement.length === 16) {
                            const m4 = new THREE.Matrix4();
                            m4.fromArray(item.placement);
                            edges.applyMatrix4(m4);
                        }
                        this.scene.add(edges);

                        this._meshes[index] = { mesh, edges, material, edgesMat };
                        loadedCount++;
                        resolve();
                    } catch (err) {
                        reject(err);
                    }
                },
                undefined,
                (err) => reject(err),
            );
        });

        // Load all in parallel
        await Promise.all(items.map((item, i) => loadOne(item, i)));

        if (this._disposed || this._loadGen !== gen) return;

        // Center all meshes
        const center = new THREE.Vector3();
        combinedBox.getCenter(center);
        this._sceneCenter.copy(center);

        for (const entry of this._meshes) {
            if (!entry) continue;
            entry.mesh.position.sub(center);
            entry.edges.position.sub(center);
        }

        // Fit camera
        this._fitCameraToBox(combinedBox, center);
    }

    /**
     * Change the color/opacity of a specific mesh by index.
     *
     * @param {number} index
     * @param {number} color  hex color
     * @param {number} [opacity=1.0]
     */
    setMeshColor(index, color, opacity = 1.0) {
        const entry = this._meshes[index];
        if (!entry) return;
        entry.material.color.setHex(color);
        entry.material.transparent = opacity < 1.0;
        entry.material.opacity = opacity;
        entry.material.needsUpdate = true;

        // Match edge visibility to mesh opacity
        if (entry.edgesMat) {
            entry.edgesMat.transparent = opacity < 1.0;
            entry.edgesMat.opacity = Math.min(opacity + 0.1, 1.0);
            entry.edgesMat.needsUpdate = true;
        }
    }

    // ── Line geometry overlay ────────────────────────────────────────

    /**
     * Draw polylines as bright line segments in the scene.
     *
     * @param {Array<Array<[number,number,number]>>} polylines
     *   Each polyline is an array of [x,y,z] points.
     * @param {number} [color=0xff3333]
     */
    addLines(polylines, color = 0xff3333) {
        if (!polylines || polylines.length === 0) return;

        const positions = [];
        for (const pts of polylines) {
            for (let i = 0; i < pts.length - 1; i++) {
                const a = pts[i], b = pts[i + 1];
                // Shift by scene center to match mesh positions
                positions.push(
                    a[0] - this._sceneCenter.x, a[1] - this._sceneCenter.y, a[2] - this._sceneCenter.z,
                    b[0] - this._sceneCenter.x, b[1] - this._sceneCenter.y, b[2] - this._sceneCenter.z,
                );
            }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

        const material = new THREE.LineBasicMaterial({
            color,
            linewidth: 2,      // Note: linewidth > 1 only works with Line2 on some backends
            depthTest: false,   // Draw on top of meshes
        });

        const lines = new THREE.LineSegments(geometry, material);
        lines.renderOrder = 999;  // render last (on top)
        this.scene.add(lines);
        this._lines.push(lines);
    }

    clearLines() {
        for (const line of this._lines) {
            this.scene.remove(line);
            line.geometry.dispose();
            line.material.dispose();
        }
        this._lines = [];
    }

    // ── Camera fitting ───────────────────────────────────────────────

    _fitCamera(geometry) {
        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        const center = new THREE.Vector3();
        box.getCenter(center);

        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);

        this._mesh.position.sub(center);
        if (this._edges) this._edges.position.sub(center);

        const fov = this.camera.fov * (Math.PI / 180);
        const dist = (maxDim / 2) / Math.tan(fov / 2) * 1.5;

        this.camera.position.set(dist * 0.7, dist * 0.5, dist);
        this.camera.near = dist / 100;
        this.camera.far = dist * 10;
        this.camera.updateProjectionMatrix();

        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    _fitCameraToBox(box, center) {
        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);

        const fov = this.camera.fov * (Math.PI / 180);
        const dist = (maxDim / 2) / Math.tan(fov / 2) * 1.5;

        this.camera.position.set(dist * 0.7, dist * 0.5, dist);
        this.camera.near = dist / 100;
        this.camera.far = dist * 10;
        this.camera.updateProjectionMatrix();

        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    // ── Cleanup helpers ──────────────────────────────────────────────

    _clearSingle() {
        if (this._mesh) {
            this.scene.remove(this._mesh);
            this._mesh.geometry.dispose();
            this._mesh.material.dispose();
            this._mesh = null;
        }
        if (this._edges) {
            this.scene.remove(this._edges);
            this._edges.geometry.dispose();
            this._edges.material.dispose();
            this._edges = null;
        }
    }

    _clearMulti() {
        for (const entry of this._meshes) {
            if (!entry) continue;
            this.scene.remove(entry.mesh);
            entry.mesh.geometry.dispose();
            entry.material.dispose();
            this.scene.remove(entry.edges);
            entry.edges.geometry.dispose();
            entry.edgesMat.dispose();
        }
        this._meshes = [];
        this.clearLines();
    }

    // ── Resize / animate / dispose ───────────────────────────────────

    _onResize() {
        if (this._disposed) return;
        const rect = this.container.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        this.width = rect.width;
        this.height = rect.height;
        this.camera.aspect = this.width / this.height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.width, this.height);
    }

    _animate() {
        if (this._disposed) return;
        this._animId = requestAnimationFrame(() => this._animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    dispose() {
        this._disposed = true;
        if (this._resizeObserver) this._resizeObserver.disconnect();
        if (this._animId) cancelAnimationFrame(this._animId);
        this._clearSingle();
        this._clearMulti();
        this.controls.dispose();
        this.renderer.dispose();
        if (this.renderer.domElement.parentNode) {
            this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
        }
    }
}
