<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\PrincipalComponentAnalysis;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;

#[Group('Transformers')]
#[CoversClass(PrincipalComponentAnalysis::class)]
class PrincipalComponentAnalysisTest extends TestCase
{
    protected PrincipalComponentAnalysis $transformer;

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

        $this->transformer = new PrincipalComponentAnalysis(2);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(PrincipalComponentAnalysis::class, $this->transformer);
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

        new PrincipalComponentAnalysis(0);
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

        $this->assertEqualsWithDelta(0.15746964404228, $this->transformer->lossiness(), 1e-8);

        $transformed = $this->dataset->samples();

        $this->transformer->transform($transformed);

        $this->assertCount(6, $transformed);
        $this->assertCount(2, $transformed[0]);

        $expected = [
            [1.8483207164864, 0.048471972345861],
            [0.92226301666678, 1.8787447846658],
            [1.4501277143791, 1.3684014009868],
            [1.9020602516136, 0.96861190148325],
            [3.1716121495566, 1.0275470793422],
            [2.5900079167171, 0.49988017816587],
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
        $fewer = new PrincipalComponentAnalysis(1);
        $fewer->fit(clone $this->dataset);

        $more = new PrincipalComponentAnalysis(2);
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

        $this->assertInstanceOf(PrincipalComponentAnalysis::class, $restored);
        $this->assertTrue($restored->fitted());
        $this->assertEquals($this->transformer->lossiness(), $restored->lossiness());

        $samples = $this->dataset->samples();
        $this->transformer->transform($samples);

        $samples2 = $this->dataset->samples();
        $restored->transform($samples2);

        $this->assertEquals($samples, $samples2);
    }
}
