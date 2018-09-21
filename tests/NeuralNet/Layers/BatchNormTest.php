<?php

namespace Rubix\Tests\NeuralNet\Layers;

use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\Other\Structures\Matrix;
use Rubix\ML\NeuralNet\Layers\Hidden;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\Layers\Parametric;
use PHPUnit\Framework\TestCase;

class BatchNormTest extends TestCase
{
    protected $fanIn;

    protected $fanOut;

    protected $input;

    protected $output;

    protected $layer;

    public function setUp()
    {
        $this->fanIn = 5;

        $this->fanOut = 5;

        $this->input = new Matrix([
            [1., 2.5, -4.],
            [0.1, 0., 2.2],
            [0.002, -6., 1.2],
        ], false);

        $this->output = [
            [0.4171398827949467, 0.9534625892455924, -1.370602472040539],
            [-0.6274558051381585, -0.7215741759088822, 1.3490299810470407],
            [0.5058265681932177, -1.3900754274119194, 0.8842488592187014],
        ];

        $this->layer = new BatchNorm(0.1);

        $this->layer->init($this->fanIn);
    }

    public function test_build_layer()
    {
        $this->assertInstanceOf(BatchNorm::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
        $this->assertInstanceOf(Hidden::class, $this->layer);
        $this->assertInstanceOf(Parametric::class, $this->layer);
    }

    public function test_width()
    {
        $this->assertEquals($this->fanOut, $this->layer->width());
    }

    public function test_forward_infer()
    {
        $out = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $out);
        $this->assertEquals($this->output, $out->asArray());

        $out = $this->layer->infer($this->input);

        $this->assertInstanceOf(Matrix::class, $out);
        $this->assertEquals($this->output, $out->asArray());
    }
}
