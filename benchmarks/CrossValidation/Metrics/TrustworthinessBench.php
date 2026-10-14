<?php

namespace Rubix\ML\Benchmarks\CrossValidation\Metrics;

use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\PrincipalComponentAnalysis;
use Rubix\ML\CrossValidation\Metrics\Trustworthiness;

/**
 * @Groups({"CrossValidation", "Metrics"})
 * @BeforeMethods({"setUp"})
 */
class TrustworthinessBench
{
    protected const SIZE = 1000;

    protected const DIMS = 10;

    protected const TARGET_DIMS = 2;

    /**
     * @var Trustworthiness
     */
    protected Trustworthiness $metric;

    /**
     * High-dimensional original samples.
     *
     * @var list<list<float>>
     */
    protected array $samples;

    /**
     * Low-dimensional embedding of $samples.
     *
     * @var list<list<float>>
     */
    protected array $embedding;

    public function setUp() : void
    {
        $center = array_fill(0, self::DIMS, 0.0);

        $dataset = (new Blob($center, 1.5))->generate(self::SIZE);

        $pca = new PrincipalComponentAnalysis(self::TARGET_DIMS);

        $this->samples = $dataset->samples();

        $dataset->apply($pca);

        $this->embedding = $dataset->samples();

        $this->metric = new Trustworthiness(5);
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("milliseconds", precision=3)
     * @return float
     */
    public function score() : float
    {
        return $this->metric->score($this->samples, $this->embedding);
    }
}
