<?php

namespace Rubix\ML\Tests\Manifold;

use Rubix\ML\Manifold\TSNE;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Manifold\Embedder;
use Rubix\ML\Kernels\Distance\Euclidean;
use PHPUnit\Framework\TestCase;

class TSNETest extends TestCase
{
    protected $embedder;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = Labeled::load(dirname(__DIR__) . '/iris.dataset')
            ->stratifiedSplit(0.2)[0];

        $this->embedder = new TSNE(2, 10, 12., 1000, 1., 0.2, 1e-6, new Euclidean(), 1e-5, 100);
    }

    public function test_build_embedder()
    {
        $this->assertInstanceOf(TSNE::class, $this->embedder);
        $this->assertInstanceOf(Embedder::class, $this->embedder);
    }

    public function test_embed()
    {
        $embedding = $this->embedder->embed($this->dataset);
        $steps = $this->embedder->steps();

        $this->assertCount(20, $embedding);

        // file_put_contents('test.json', json_encode($steps, JSON_PRETTY_PRINT));
    }
}
