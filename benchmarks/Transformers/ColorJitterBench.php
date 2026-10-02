<?php

namespace Rubix\ML\Benchmarks\Transformers;

use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\ColorJitter;

/**
 * @Groups({"Transformers"})
 */
class ColorJitterBench
{
    /**
     * @var Unlabeled
     */
    protected Unlabeled $dataset;

    /**
     * @var ColorJitter
     */
    protected ColorJitter $transformer;

    public function setUp() : void
    {
        // Create a small dataset with images
        $samples = [];

        for ($i = 0; $i < 10; ++$i) {
            $image = imagecreatetruecolor(100, 100);
            imagefill($image, 0, 0, imagecolorallocate($image, 100, 150, 200));
            $samples[] = [$image];
        }

        $this->dataset = new Unlabeled($samples);

        $this->transformer = new ColorJitter(0.2, 0.2, 0.2, 30.0);
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
