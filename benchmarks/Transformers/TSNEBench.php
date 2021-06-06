<?php

namespace Rubix\ML\Benchmarks\Embedders;

use Rubix\ML\Transformers\TSNE;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;

/**
 * @Groups({"Transformers"})
 * @BeforeMethods({"setUp"})
 */
class TSNEBench
{
    protected const TESTING_SIZE = 1000;

    /**
     * @var \Rubix\ML\Datasets\Labeled;
     */
    protected $testing;

    /**
     * @var \Rubix\ML\Transformers\TSNE
     */
    protected $embedder;

    public function setUp() : void
    {
        $generator = new Agglomerate([
            'Iris-setosa' => new Blob([5.0, 3.42, 1.46, 0.24], [0.35, 0.38, 0.17, 0.1]),
            'Iris-versicolor' => new Blob([5.94, 2.77, 4.26, 1.33], [0.51, 0.31, 0.47, 0.2]),
            'Iris-virginica' => new Blob([6.59, 2.97, 5.55, 2.03], [0.63, 0.32, 0.55, 0.27]),
        ]);

        $this->testing = $generator->generate(self::TESTING_SIZE);

        $this->embedder = new TSNE(1);
    }

    /**
     * @Subject
     * @Skip
     * @Iterations(5)
     * @OutputTimeUnit("seconds", precision=3)
     */
    public function apply() : void
    {
        $this->testing->apply($this->embedder);
    }
}
