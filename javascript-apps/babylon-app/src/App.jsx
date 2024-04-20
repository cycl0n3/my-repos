import React from "react";

import * as BABYLON from '@babylonjs/core';

import '@babylonjs/loaders';

import { Assets } from '@babylonjs/assets';

import SceneComponent from "./components/SceneComponent"; // uses above component in same directory

// import SceneComponent from 'babylonjs-hook'; // if you install 'babylonjs-hook' NPM.

import "./App.css";

let box;

const onSceneReady = (scene) => {
  // This creates and positions a free camera (non-mesh)
  const camera = new BABYLON.FreeCamera("camera1", new BABYLON.Vector3(0, 5, -10), scene);

  // This targets the camera to scene origin
  camera.setTarget(BABYLON.Vector3.Zero());

  const canvas = scene.getEngine().getRenderingCanvas();

  // This attaches the camera to the canvas
  camera.attachControl(canvas, true);

  // This creates a light, aiming 0,1,0 - to the sky (non-mesh)
  const light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 1), scene);

  // Default intensity is 1. Let's dim the light a small amount
  light.intensity = 0.9;

  const box = BABYLON.MeshBuilder.CreateBox("box", {});
  box.position.y = 0.5;

  const roof = BABYLON.MeshBuilder.CreateCylinder("roof", {diameter: 1.3, height: 1.2, tessellation: 3});
  roof.scaling.x = 0.75;
  roof.rotation.z = Math.PI / 2;
  roof.position.y = 1.22;

  // Our built-in 'ground' shape.
  const ground = BABYLON.MeshBuilder.CreateGround("ground", {width: 10, height: 10});

  const groundMaterial = new BABYLON.StandardMaterial("Ground Material", scene);
  groundMaterial.diffuseColor = BABYLON.Color3.Red();

  let groundTexture = new BABYLON.Texture(Assets.textures.checkerboard_basecolor_png.rootUrl, scene);
  groundMaterial.diffuseTexture = groundTexture;
  
  ground.material = groundMaterial;
};

/**
 * Will run on every frame render.  We are spinning the box on y-axis.
 */
const onRender = (scene) => {
  if (box !== undefined) {
    const deltaTimeInMillis = scene.getEngine().getDeltaTime();

    const rpm = 15;
    box.rotation.y += (rpm / 60) * Math.PI * 2 * (deltaTimeInMillis / 1000);
  }
};

export default () => (
  <div>
    <SceneComponent antialias onSceneReady={onSceneReady} onRender={onRender} id="my-canvas" />
  </div>
);
