
// var game = new Phaser.Game(800, 600, Phaser.CANVAS, 'phaser-example', { preload: preload, create: create, update: update });
var game = new Phaser.Game(800, 600, Phaser.AUTO, 'phaser-example', { preload: preload, create: create, update: update });

function preload() {

    game.load.tilemap('mario', 'assets/tilemaps/maps/super_mario.json', null, Phaser.Tilemap.TILED_JSON);
    game.load.image('tiles', 'assets/tilemaps/tiles/super_mario.png');
    game.load.image('player', 'assets/sprites/phaser-dude.png');

}

var map;
var tileset;
var layer;
var layer2;
var p;
var cursors;

function create() {

    game.stage.backgroundColor = '#787878';

    map = game.add.tilemap('mario');

    map.addTilesetImage('SuperMarioBros-World1-1', 'tiles');

    map.setCollisionBetween(15, 16);
    map.setCollisionBetween(20, 25);
    map.setCollisionBetween(27, 29);
    map.setCollision(40);
    
    layer = map.createLayer('World1', 600);
    layer.x = 132;
    layer.y = 48;

    layer2 = map.createLayer('World1', 600);
    layer2.x = 32;
    layer2.y = 316;

    game.world.setBounds(0, 0, layer.widthInPixels * layer.scale.x, 600);

    p = game.add.sprite(128, 200, 'player');

    game.physics.arcade.enable(p);

    game.physics.arcade.gravity.y = 300;

    p.body.linearDamping = 1;
    p.body.collideWorldBounds = true;

    game.camera.follow(p);

    cursors = game.input.keyboard.createCursorKeys();

}

function update() {

    game.physics.arcade.collide(p, layer);
    game.physics.arcade.collide(p, layer2);

    p.body.velocity.x = 0;

    if (cursors.up.isDown && p.body.onFloor())
    {
        p.body.velocity.y = -300;
    }

    if (cursors.left.isDown)
    {
        p.body.velocity.x = -150;
    }
    else if (cursors.right.isDown)
    {
        p.body.velocity.x = 150;
    }

}
