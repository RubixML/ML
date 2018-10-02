<?php

namespace Rubix\Tests\Manifold;

use Rubix\ML\Manifold\TSNE;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Manifold\Embedder;
use Rubix\ML\Kernels\Distance\Manhattan;
use PHPUnit\Framework\TestCase;

class TSNETest extends TestCase
{
    protected $embedder;

    protected $samples;

    public function setUp()
    {
        $this->samples = Labeled::restore(dirname(__DIR__) . '/iris.dataset');

        $this->embedder = new TSNE(2, 30, 12., 1000, 1., 0.2, 1e-6, new Manhattan(), 1e-5, 100);
    }

    public function test_build_embedder()
    {
        $this->assertInstanceOf(TSNE::class, $this->embedder);
        $this->assertInstanceOf(Embedder::class, $this->embedder);
    }
}
