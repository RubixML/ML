<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Transformers;

use Rubix\ML\DataType;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\Stateful;
use Rubix\ML\Transformers\Transformer;
use Rubix\ML\Transformers\LinearDiscriminantAnalysis;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\TestCase;
use Rubix\ML\CrossValidation\Metrics\Trustworthiness;

#[Group('Transformers')]
#[CoversClass(LinearDiscriminantAnalysis::class)]
class LinearDiscriminantAnalysisTest extends TestCase
{
    /**
     * The number of samples in the validation set.
     *
     * @var int
     */
    protected const TEST_SIZE = 64;

    /**
     * The minimum trustworthiness score required to pass the test.
     */
    protected const float MIN_SCORE = 0.8;

    /**
     * Constant used to seed the random number generator.
     *
     * @var int
     */
    protected const RANDOM_SEED = 0;

    protected LinearDiscriminantAnalysis $transformer;

    protected Labeled $dataset;

    protected function setUp() : void
    {
        $this->dataset = new Labeled(
            samples: [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 1.0],
                [4.0, 1.0, 3.0],
                [1.1, 2.1, 2.9],
                [2.1, 3.9, 1.1],
                [3.9, 1.1, 3.1],
            ],
            labels: ['red', 'green', 'blue', 'red', 'green', 'blue']
        );

        $this->transformer = new LinearDiscriminantAnalysis(1);
    }

    #[Test]
    public function build() : void
    {
        $this->assertInstanceOf(LinearDiscriminantAnalysis::class, $this->transformer);
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

        new LinearDiscriminantAnalysis(0);
    }

    #[Test]
    public function compatibility() : void
    {
        $this->assertEquals([DataType::continuous()], $this->transformer->compatibility());
    }

    #[Test]
    public function fitUnlabeledThrows() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $dataset = new Unlabeled(samples: [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 1.0],
            [4.0, 1.0, 3.0],
        ]);

        $this->transformer->fit($dataset);
    }

    #[Test]
    public function fitContinuousLabelsThrows() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $dataset = new Labeled(
            samples: [[1.0, 2.0, 3.0], [2.0, 4.0, 1.0], [4.0, 1.0, 3.0]],
            labels: [0.1, 0.2, 0.3]
        );

        $this->transformer->fit($dataset);
    }

    #[Test]
    public function fitTransform() : void
    {
        $this->assertEquals(3, $this->dataset->numFeatures());
        $this->assertCount(3, $this->dataset->possibleOutcomes());

        $this->transformer->fit($this->dataset);

        $this->assertTrue($this->transformer->fitted());
        $this->assertEqualsWithDelta(0.28387957837822, $this->transformer->lossiness(), 1e-8);

        $transformed = $this->dataset->samples();

        $this->transformer->transform($transformed);

        $this->assertCount(6, $transformed);
        $this->assertCount(1, $transformed[0]);

        $expected = [
            [0.52560869955198],
            [1.3441218195234],
            [2.7768203252381],
            [0.45777094802498],
            [1.1736889702894],
            [2.7022010447491],
        ];

        $magnitudes = array_map(
            static fn (array $row) => array_map('abs', $row),
            $transformed
        );

        $this->assertEqualsWithDelta($expected, $magnitudes, 1e-8);
    }

    #[Test]
    public function trustworthiness() : void
    {
        srand(self::RANDOM_SEED);

        $generator = new Agglomerate([
            'red' => new Blob([255, 32, 0], 30.0),
            'green' => new Blob([0, 128, 0], 10.0),
            'blue' => new Blob([0, 32, 255], 20.0),
        ], [2, 3, 4]);

        $dataset = $generator->generate(self::TEST_SIZE);

        $original = clone $dataset;

        $reduction = new LinearDiscriminantAnalysis(1);

        $reduction->fit($dataset);

        $transformed = $dataset->samples();

        $reduction->transform($transformed);

        $score = (new Trustworthiness())->score($original->samples(), $transformed);

        $this->assertGreaterThanOrEqual(self::MIN_SCORE, $score);
    }

    #[Test]
    public function lossinessRetainsMoreVarianceForMoreDimensions() : void
    {
        $fewer = new LinearDiscriminantAnalysis(1);
        $fewer->fit(clone $this->dataset);

        $more = new LinearDiscriminantAnalysis(2);
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

        $this->assertInstanceOf(LinearDiscriminantAnalysis::class, $restored);
        $this->assertTrue($restored->fitted());
        $this->assertEquals($this->transformer->lossiness(), $restored->lossiness());

        $samples = $this->dataset->samples();
        $this->transformer->transform($samples);

        $samples2 = $this->dataset->samples();
        $restored->transform($samples2);

        $this->assertEquals($samples, $samples2);
    }
}
