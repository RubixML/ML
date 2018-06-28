<?php

namespace Rubix\Tests\NeuralNet;

use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Softmax;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;
use PHPUnit\Framework\TestCase;

class NetworkTest extends TestCase
{
    protected $network;

    public function setUp()
    {
        $this->network = new Network(new Input(5), [
            new Dense(5, new ELU()),
            new Dense(5, new ELU()),
        ], new Softmax(['yes', 'no', 'maybe']), new Adam(0.001));
    }

    public function test_build_network()
    {
        $this->assertInstanceOf(Network::class, $this->network);
    }

    public function test_depth()
    {
        $this->assertEquals(3, $this->network->depth());
    }
}
