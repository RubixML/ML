<?php

namespace Rubix\ML\Benchmarks\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\ImageVectorizer;
use Rubix\ML\Datasets\Dataset;

/**
 * @Groups({"Transformers"})
 * @BeforeMethods({"setUp"})
 */
class ImageVectorizerBench
{
    protected const DATASET_SIZE = 1000;

    /**
     * @var Dataset
     */
    public Dataset $dataset;

    /**
     * @var ImageVectorizer
     */
    protected ImageVectorizer $transformer;

    public function setUp() : void
    {
        $samples = [];

        for ($i = 0; $i < self::DATASET_SIZE; ++$i) {
            $samples[] = [imagecreatefrompng('tests/test.png')];
        }

        $this->dataset = Unlabeled::build($samples);

        $this->transformer = new ImageVectorizer();
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("seconds", precision=3)
     */
    public function apply() : void
    {
        $this->dataset->apply($this->transformer);
    }
}
