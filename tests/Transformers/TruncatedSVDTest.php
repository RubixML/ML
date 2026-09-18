<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\TruncatedSVD;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(TruncatedSVD::class)]
class TruncatedSVDTest extends TestCase
{
    protected TruncatedSVD $transformer;

    protected Unlabeled $dataset;

    protected function setUp() : void
    {
        $this->dataset = new Unlabeled(samples: [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 1.0, 3.0],
            [4.0, 1.0, 3.0, 2.0],
            [3.0, 3.0, 4.0, 1.0],
            [1.0, 1.0, 1.0, 5.0],
            [5.0, 2.0, 2.0, 1.0],
        ]);

        $this->transformer = new TruncatedSVD(2);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(TruncatedSVD::class, $this->transformer);
        $this->assertInstanceOf(Transformer::class, $this->transformer);
        $this->assertInstanceOf(Stateful::class, $this->transformer);
        $this->assertInstanceOf(Persistable::class, $this->transformer);
        $this->assertNotTrue($this->transformer->fitted());
        $this->assertNull($this->transformer->lossiness());
    }

    #[Test]
    public function dimensionsBelowOneThrows() : void
    {
        $this->expectException(InvalidArgumentException::class);

        new TruncatedSVD(0);
    }

    #[Test]
    public function compatibility() : void
    {
        $this->assertEquals([DataType::continuous()], $this->transformer->compatibility());
    }

    #[Test]
    public function fitIncompatibleDataThrows() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $dataset = new Unlabeled(samples: [[1, 2, 3, 4], [2, 4, 1, 3], [4, 1, 3, 2]]);

        $this->transformer->fit(clone $dataset);
    }

    #[Test]
    public function fitTransform() : void
    {
        $this->assertEquals(4, $this->dataset->numFeatures());

        $this->transformer->fit($this->dataset);

        $this->assertTrue($this->transformer->fitted());

        $this->assertEqualsWithDelta(0.23296036563469, $this->transformer->lossiness(), 1e-8);

        $transformed = $this->dataset->samples();

        $this->transformer->transform($transformed);

        $this->assertCount(6, $transformed);
        $this->assertCount(2, $transformed[0]);

        $expected = [
            [4.9457549806412, 2.0359554370691],
            [4.904019417135, 1.0866274691921],
            [5.1326733170795, 1.2958866953735],
            [5.4375322351481, 1.6773917861347],
            [4.0479936458723, 3.2050408632791],
            [5.1312849289466, 2.4555114759292],
        ];

        $magnitudes = array_map(
            static fn (array $row) => array_map('abs', $row),
            $transformed
        );

        $this->assertEqualsWithDelta($expected, $magnitudes, 1e-8);
    }

    #[Test]
    public function lossinessRetainsMoreVarianceForMoreDimensions() : void
    {
        $fewer = new TruncatedSVD(1);
        $fewer->fit(clone $this->dataset);

        $more = new TruncatedSVD(2);
        $more->fit(clone $this->dataset);

        $this->assertGreaterThan(
            $more->lossiness(),
            $fewer->lossiness(),
            'Fewer retained dimensions should lose more variance.'
        );
    }

    #[Test]
    public function transformUnfittedThrows() : void
    {
        $this->expectException(RuntimeException::class);

        $samples = $this->dataset->samples();

        $this->transformer->transform($samples);
    }

    #[Test]
    public function restoreStateFromSerializedModel() : void
    {
        $this->transformer->fit($this->dataset);

        $restored = unserialize(serialize($this->transformer));

        $this->assertInstanceOf(TruncatedSVD::class, $restored);
        $this->assertTrue($restored->fitted());
        $this->assertEquals($this->transformer->lossiness(), $restored->lossiness());

        $samples = $this->dataset->samples();
        $this->transformer->transform($samples);

        $samples2 = $this->dataset->samples();
        $restored->transform($samples2);

        $this->assertEquals($samples, $samples2);
    }
}
