<?php

namespace Rubix\ML\Benchmarks\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\OneHotEncoder;

/**
 * @Groups({"Transformers"})
 * @BeforeMethods({"setUp"})
 */
class OneHotEncoderBench
{
    protected const DATASET_SIZE = 2500;

    protected const NUM_COLUMNS = 3;

    protected const CATEGORIES = ['red', 'green', 'blue', 'yellow', 'purple', 'orange'];

    /**
     * @var \Rubix\ML\Datasets\Dataset
     */
    public $dataset;

    /**
     * @var OneHotEncoder
     */
    protected $transformer;

    public function setUp() : void
    {
        $categories = self::CATEGORIES;

        $numCategories = count($categories);

        $samples = [];

        for ($i = 0; $i < self::DATASET_SIZE; ++$i) {
            $sample = [];

            for ($j = 0; $j < self::NUM_COLUMNS; ++$j) {
                $sample[] = $categories[($i + $j) % $numCategories];
            }

            $samples[] = $sample;
        }

        $this->dataset = Unlabeled::quick($samples);

        $this->transformer = new OneHotEncoder();
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function apply() : void
    {
        $this->dataset->apply($this->transformer);
    }
}
