<?php

namespace Rubix\ML\Benchmarks\Transformers;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\TargetEncoder;

use function count;

/**
 * @Groups({"Transformers"})
 * @BeforeMethods({"setUp"})
 */
class TargetEncoderBench
{
    protected const DATASET_SIZE = 2500;

    protected const NUM_COLUMNS = 3;

    protected const CATEGORIES = ['red', 'green', 'blue', 'yellow', 'purple', 'orange'];

    /**
     * @var Labeled
     */
    public Labeled $dataset;

    /**
     * @var TargetEncoder
     */
    protected TargetEncoder $transformer;

    public function setUp() : void
    {
        $categories = self::CATEGORIES;

        $numCategories = count($categories);

        $samples = $labels = [];

        for ($i = 0; $i < self::DATASET_SIZE; ++$i) {
            $sample = [];

            for ($j = 0; $j < self::NUM_COLUMNS; ++$j) {
                $sample[] = $categories[($i + $j) % $numCategories];
            }

            $samples[] = $sample;

            $labels[] = (float) $i;
        }

        $this->dataset = Labeled::quick($samples, $labels);

        $this->transformer = new TargetEncoder();
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function apply() : void
    {
        $transformer = new TargetEncoder();

        $this->dataset->apply($transformer);
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function applyHeavilySmoothed() : void
    {
        $transformer = new TargetEncoder(smoothing: 100.0);

        $this->dataset->apply($transformer);
    }
}
