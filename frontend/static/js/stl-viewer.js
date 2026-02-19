/**
 * Three.js STL viewer component - reusable 3D preview that fills its container
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

export class STLViewer {
    constructor(containerEl) {
        this.container = containerEl;
        this._disposed = false;

        // Read actual container size (must have explicit CSS dimensions)
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

        this._mesh = null;
        this._animId = null;
        this._loadGen = 0;  // incremented on each loadSTL call; used to cancel stale loads

        // Resize observer to keep renderer matched to container
        this._resizeObserver = new ResizeObserver(() => this._onResize());
        this._resizeObserver.observe(containerEl);

        this._animate();
    }

    async loadSTL(url) {
        // Each call gets a unique generation number.  If the viewer is disposed
        // or a newer load starts before this one completes, we reject rather than
        // silently hanging (which leaves the panel stuck in "loading" state).
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
                        // Remove previous mesh
                        if (this._mesh) {
                            this.scene.remove(this._mesh);
                            this._mesh.geometry.dispose();
                            this._mesh.material.dispose();
                        }

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

    _fitCamera(geometry) {
        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        const center = new THREE.Vector3();
        box.getCenter(center);

        const size = new THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);

        // Center geometry
        this._mesh.position.sub(center);

        // Position camera
        const fov = this.camera.fov * (Math.PI / 180);
        const dist = (maxDim / 2) / Math.tan(fov / 2) * 1.5;

        this.camera.position.set(dist * 0.7, dist * 0.5, dist);
        this.camera.near = dist / 100;
        this.camera.far = dist * 10;
        this.camera.updateProjectionMatrix();

        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

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
        if (this._mesh) {
            this.scene.remove(this._mesh);
            this._mesh.geometry.dispose();
            this._mesh.material.dispose();
        }
        this.controls.dispose();
        this.renderer.dispose();
        if (this.renderer.domElement.parentNode) {
            this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
        }
    }
}
