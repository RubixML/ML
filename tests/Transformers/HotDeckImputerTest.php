<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Transformers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\HotDeckImputer;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(HotDeckImputer::class)]
class HotDeckImputerTest extends TestCase
{
    protected const int RANDOM_SEED = 0;

    protected Blob $generator;

    protected HotDeckImputer $transformer;

    protected function setUp() : void
    {
        $this->generator = new Blob(center: [30.0, 0.0]);

        $this->transformer = new HotDeckImputer(k: 2, weighted: true, categoricalPlaceholder: '?');

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function fitTransform() : void
    {
        $dataset = new Unlabeled(samples: [
            [30.0, 0.001],
            [NAN, 0.055],
            [50.0, -2.0],
            [60.0, NAN],
            [10.0, 1.0],
            [100.0, 9.0],
        ]);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $dataset->apply($this->transformer);

        $this->assertEquals(10, $dataset[1][0]);
        $this->assertEquals(0.001, $dataset[3][1]);
    }

    #[Test]
    public function restoreStateFromSerializedModel() : void
    {
        $dataset = new Unlabeled(samples: [
            [30.0, 0.001],
            [NAN, 0.055],
            [50.0, -2.0],
            [60.0, NAN],
            [10.0, 1.0],
            [100.0, 9.0],
        ]);

        $this->transformer->fit($dataset);

        $this->assertTrue($this->transformer->fitted());

        $restored = unserialize(serialize($this->transformer));

        $this->assertTrue($restored->fitted());
    }
}
