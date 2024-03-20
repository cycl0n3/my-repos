import { useEffect, useState } from "react";

import "./App.css";

import Phaser from "phaser";

class GameScene extends Phaser.Scene {
  
  platforms;
  player;

  preload() {
    this.load.setBaseURL("https://labs.phaser.io");

    this.load.image('sky', 'assets/skies/sky3.png');
    this.load.image('ground', 'assets/sprites/platform.png');
    this.load.image('star', 'assets/sprites/star.png');
    this.load.image('bomb', 'assets/sprites/bomb.png');
    this.load.spritesheet('dude',
        'assets/sprites/dude.png',
        { frameWidth: 32, frameHeight: 48 }
    );
  }

  create() {
    this.add.image(400, 300, "sky");
    // this.add.image(400, 300, 'star');
    // this.add.image(400, 300, 'dude');

    this.platforms = this.physics.add.staticGroup();

    this.platforms.create(400, 568, 'ground').setScale(2).refreshBody();
    this.platforms.create(600, 400, 'ground');
    this.platforms.create(50, 250, 'ground');
    this.platforms.create(750, 220, 'ground');

    this.player = this.physics.add.sprite(100, 450, 'dude');
    this.player.setBounce(1);
    this.player.setCollideWorldBounds(true);
  }

  update() {

  }
}

function App() {
  useEffect(() => {
    const game = new Phaser.Game({
      type: Phaser.AUTO,
      width: 800,
      height: 600,
      scene: GameScene,
      parent: 'game-container',
      physics: {
        default: "arcade",
        arcade: {
          gravity: { y: 300 },
        },
      },
    });

    return () => {
      // Clean up Phaser game
      game.destroy(true);
    };
  }, []);

  return <div id="game-container" />;
}

export default App;
