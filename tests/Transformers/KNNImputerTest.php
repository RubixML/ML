<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\KNNImputer;
use Rubix\ML\Datasets\Generators\Blob;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(KNNImputer::class)]
class KNNImputerTest extends TestCase
{
    protected const int RANDOM_SEED = 0;

    protected Blob $generator;

    protected KNNImputer $transformer;

    protected function setUp() : void
    {
        $this->generator = new Blob(center: [30.0, 0.0]);

        $this->transformer = new KNNImputer(k: 2, weighted: true, categoricalPlaceholder: '?');

        srand(self::RANDOM_SEED);
    }

    public function testFitTransform() : void
    {
        $dataset = new Unlabeled(samples: [
            [30, 0.001],
            [NAN, 0.055],
            [50, -2.0],
            [60, NAN],
            [10, 1.0],
            [100, 9.0],
        ]);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $dataset->apply($this->transformer);

        $this->assertEquals(23.692172188539388, $dataset[1][0]);
        $this->assertEquals(-1.4826674509492581, $dataset[3][1]);
    }
}
