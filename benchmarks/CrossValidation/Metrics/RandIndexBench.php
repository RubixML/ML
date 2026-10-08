<?php

namespace Rubix\ML\Benchmarks\CrossValidation\Metrics;

use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\CrossValidation\Metrics\RandIndex;

use function random_int;

/**
 * @Groups({"CrossValidation", "Metrics"})
 * @BeforeMethods({"setUp"})
 */
class RandIndexBench
{
    protected const SIZE = 100000;

    protected const CLUSTERS = 50;

    protected const LABELS = 100;

    /**
     * @var RandIndex
     */
    protected RandIndex $metric;

    /**
     * @var list<int>
     */
    protected array $predictions;

    /**
     * @var list<int>
     */
    protected array $labels;

    public function setUp() : void
    {
        $dataset = (new Blob([0.0, 0.0], [1.0, 1.0]))->generate(self::SIZE);

        $estimator = new KMeans(self::CLUSTERS);
        $estimator->train($dataset);

        $this->predictions = $estimator->predict($dataset);

        $this->labels = [];

        for ($i = 0; $i < self::SIZE; ++$i) {
            $this->labels[] = random_int(0, self::LABELS);
        }

        $this->metric = new RandIndex();
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("milliseconds", precision=3)
     * @return float
     */
    public function score() : float
    {
        return $this->metric->score($this->predictions, $this->labels);
    }
}
