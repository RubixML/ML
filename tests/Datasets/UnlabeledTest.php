<?php

namespace Rubix\ML\Tests\Datasets;

use Rubix\ML\Report;
use Rubix\ML\DataType;
use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Datasets\Unlabeled;
use PHPUnit\Framework\TestCase;
use IteratorAggregate;
use ArrayAccess;
use Countable;

use function Rubix\ML\array_transpose;

/**
 * @group Datasets
 * @covers \Rubix\ML\Datasets\Unlabeled
 */
class UnlabeledTest extends TestCase
{
    protected const SAMPLES = [
        ['nice', 'furry', 'friendly', 4.0],
        ['mean', 'furry', 'loner', -1.5],
        ['nice', 'rough', 'friendly', 2.6],
        ['mean', 'rough', 'friendly', -1.0],
        ['nice', 'rough', 'friendly', 2.9],
        ['nice', 'furry', 'loner', -5.0],
    ];

    protected const TYPES = [
        DataType::CATEGORICAL,
        DataType::CATEGORICAL,
        DataType::CATEGORICAL,
        DataType::CONTINUOUS,
    ];

    protected const WEIGHTS = [
        1, 1, 2, 1, 2, 3,
    ];

    protected const RANDOM_SEED = 0;

    /**
     * @var Unlabeled
     */
    protected $dataset;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataset = new Unlabeled(self::SAMPLES, false);

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Unlabeled::class, $this->dataset);
        $this->assertInstanceOf(Dataset::class, $this->dataset);
        $this->assertInstanceOf(Countable::class, $this->dataset);
        $this->assertInstanceOf(ArrayAccess::class, $this->dataset);
        $this->assertInstanceOf(IteratorAggregate::class, $this->dataset);
    }

    /**
     * @test
     */
    public function fromIterator() : void
    {
        $dataset = Unlabeled::fromIterator(new NDJSON('tests/test.ndjson'));

        $this->assertInstanceOf(Unlabeled::class, $dataset);
    }

    /**
     * @test
     */
    public function stack() : void
    {
        $dataset1 = new Unlabeled([['sample1']]);
        $dataset2 = new Unlabeled([['sample2']]);
        $dataset3 = new Unlabeled([['sample3']]);

        $dataset = Unlabeled::stack([$dataset1, $dataset2, $dataset3]);

        $this->assertInstanceOf(Unlabeled::class, $dataset);

        $this->assertEquals(3, $dataset->numSamples());
        $this->assertEquals(1, $dataset->numFeatures());
    }

    /**
     * @test
     */
    public function samples() : void
    {
        $this->assertEquals(self::SAMPLES, $this->dataset->samples());
    }

    /**
     * @test
     */
    public function sample() : void
    {
        $this->assertEquals(self::SAMPLES[2], $this->dataset->sample(2));
        $this->assertEquals(self::SAMPLES[5], $this->dataset->sample(5));
    }

    /**
     * @test
     */
    public function numSamples() : void
    {
        $this->assertEquals(6, $this->dataset->numSamples());
    }

    /**
     * @test
     */
    public function feature() : void
    {
        $expected = array_column(self::SAMPLES, 2);

        $this->assertEquals($expected, $this->dataset->feature(2));
    }

    /**
     * @test
     */
    public function dropFeature() : void
    {
        $expected = [
            ['nice', 'friendly', 4.0],
            ['mean', 'loner', -1.5],
            ['nice', 'friendly', 2.6],
            ['mean', 'friendly', -1.0],
            ['nice', 'friendly', 2.9],
            ['nice', 'loner', -5.0],
        ];

        $this->dataset->dropFeature(1);

        $this->assertEquals($expected, $this->dataset->samples());
    }

    /**
     * @test
     */
    public function numFeatures() : void
    {
        $this->assertEquals(4, $this->dataset->numFeatures());
    }

    /**
     * @test
     */
    public function featureType() : void
    {
        $this->assertEquals(DataType::categorical(), $this->dataset->featureType(0));
        $this->assertEquals(DataType::categorical(), $this->dataset->featureType(1));
        $this->assertEquals(DataType::categorical(), $this->dataset->featureType(2));
        $this->assertEquals(DataType::continuous(), $this->dataset->featureType(3));
    }

    /**
     * @test
     */
    public function featureTypes() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::categorical(),
            DataType::categorical(),
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->dataset->featureTypes());
    }

    /**
     * @test
     */
    public function uniqueTypes() : void
    {
        $this->assertCount(2, $this->dataset->uniqueTypes());
    }

    /**
     * @test
     */
    public function homogeneous() : void
    {
        $this->assertFalse($this->dataset->homogeneous());
    }

    /**
     * @test
     */
    public function shape() : void
    {
        $this->assertEquals([6, 4], $this->dataset->shape());
    }

    /**
     * @test
     */
    public function size() : void
    {
        $this->assertEquals(24, $this->dataset->size());
    }

    /**
     * @test
     */
    public function features() : void
    {
        $expected = array_transpose(self::SAMPLES);

        $this->assertEquals($expected, $this->dataset->features());
    }

    /**
     * @test
     */
    public function types() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::categorical(),
            DataType::categorical(),
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->dataset->types());
    }

    /**
     * @test
     */
    public function filter() : void
    {
        $isFriendly = function ($record) {
            return $record[2] === 'friendly';
        };

        $filtered = $this->dataset->filter($isFriendly);

        $expected = [
            ['nice', 'furry', 'friendly', 4.0],
            ['nice', 'rough', 'friendly', 2.6],
            ['mean', 'rough', 'friendly', -1.0],
            ['nice', 'rough', 'friendly', 2.9],
        ];

        $this->assertEquals($expected, $filtered->samples());
    }

    /**
     * @test
     */
    public function sort() : void
    {
        $dataset = $this->dataset->sort(function ($recordA, $recordB) {
            return $recordA[3] > $recordB[3];
        });

        $expected = [
            ['nice', 'furry', 'loner', -5.0],
            ['mean', 'furry', 'loner', -1.5],
            ['mean', 'rough', 'friendly', -1.0],
            ['nice', 'rough', 'friendly', 2.6],
            ['nice', 'rough', 'friendly', 2.9],
            ['nice', 'furry', 'friendly', 4.0],
        ];

        $this->assertEquals($expected, $dataset->samples());
    }

    /**
     * @test
     */
    public function featuresByType() : void
    {
        $expected = array_slice(array_transpose(self::SAMPLES), 0, 3);

        $columns = $this->dataset->featuresByType(DataType::categorical());

        $this->assertEquals($expected, $columns);
    }

    /**
     * @test
     */
    public function empty() : void
    {
        $this->assertFalse($this->dataset->empty());
    }

    /**
     * @test
     */
    public function randomize() : void
    {
        $samples = $this->dataset->samples();

        $this->dataset->randomize();

        $this->assertNotEquals($samples, $this->dataset->samples());
    }

    /**
     * @test
     */
    public function head() : void
    {
        $subset = $this->dataset->head(3);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    /**
     * @test
     */
    public function tail() : void
    {
        $subset = $this->dataset->tail(3);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    /**
     * @test
     */
    public function take() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->take(3);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(3, $subset);
        $this->assertCount(3, $this->dataset);
    }

    /**
     * @test
     */
    public function leave() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->leave(1);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(5, $subset);
        $this->assertCount(1, $this->dataset);
    }

    /**
     * @test
     */
    public function slice() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->slice(2, 2);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(2, $subset);
        $this->assertCount(6, $this->dataset);
    }

    /**
     * @test
     */
    public function splice() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->splice(2, 2);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(2, $subset);
        $this->assertCount(4, $this->dataset);
    }

    /**
     * @test
     */
    public function split() : void
    {
        [$left, $right] = $this->dataset->split(0.5);

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    /**
     * @test
     */
    public function fold() : void
    {
        $folds = $this->dataset->fold(2);

        $this->assertCount(2, $folds);
        $this->assertCount(3, $folds[0]);
        $this->assertCount(3, $folds[1]);
    }

    /**
     * @test
     */
    public function batch() : void
    {
        $batches = $this->dataset->batch(2);

        $this->assertCount(3, $batches);
        $this->assertCount(2, $batches[0]);
        $this->assertCount(2, $batches[1]);
        $this->assertCount(2, $batches[2]);
    }

    /**
     * @test
     */
    public function partition() : void
    {
        [$left, $right] = $this->dataset->splitByFeature(2, 'loner');

        $this->assertInstanceOf(Unlabeled::class, $left);
        $this->assertInstanceOf(Unlabeled::class, $right);

        $this->assertCount(2, $left);
        $this->assertCount(4, $right);
    }

    /**
     * @test
     */
    public function randomSubset() : void
    {
        $subset = $this->dataset->randomSubset(3);

        $this->assertCount(3, array_unique($subset->samples(), SORT_REGULAR));
    }

    /**
     * @test
     */
    public function randomSubsetWithReplacement() : void
    {
        $subset = $this->dataset->randomSubsetWithReplacement(3);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    /**
     * @test
     */
    public function randomWeightedSubsetWithReplacement() : void
    {
        $subset = $this->dataset->randomWeightedSubsetWithReplacement(3, self::WEIGHTS);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    /**
     * @test
     */
    public function merge() : void
    {
        $this->assertCount(count(self::SAMPLES), $this->dataset);

        $dataset = new Unlabeled([['nice', 'furry', 'friendly', 4.7]]);

        $merged = $this->dataset->merge($dataset);

        $this->assertCount(count(self::SAMPLES) + 1, $merged);

        $this->assertEquals(['nice', 'furry', 'friendly', 4.7], $merged->sample(6));
    }

    /**
     * @test
     */
    public function join() : void
    {
        $this->assertEquals(count(current(self::SAMPLES)), $this->dataset->numFeatures());

        $dataset = new Unlabeled([
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
        ]);

        $joined = $this->dataset->join($dataset);

        $this->assertEquals(count(current(self::SAMPLES)) + 1, $joined->numFeatures());

        $this->assertEquals(['mean', 'furry', 'loner', -1.5, 2], $joined->sample(1));
        $this->assertEquals(['nice', 'rough', 'friendly', 2.6, 3], $joined->sample(2));
    }

    /**
     * @test
     */
    public function describe() : void
    {
        $results = $this->dataset->describe();

        $expected = [
            [
                'offset' => 0,
                'type' => 'categorical',
                'num categories' => 2,
                'probabilities' => [
                    'nice' => 0.6666666666666666,
                    'mean' => 0.3333333333333333,
                ],
            ],
            [
                'offset' => 1,
                'type' => 'categorical',
                'num categories' => 2,
                'probabilities' => [
                    'furry' => 0.5,
                    'rough' => 0.5,
                ],
            ],
            [
                'offset' => 2,
                'type' => 'categorical',
                'num categories' => 2,
                'probabilities' => [
                    'friendly' => 0.6666666666666666,
                    'loner' => 0.3333333333333333,
                ],
            ],
            [
                'offset' => 3,
                'type' => 'continuous',
                'mean' => 0.3333333333333333,
                'variance' => 9.792222222222222,
                'standard deviation' => 3.129252661934191,
                'skewness' => -0.4481030843690633,
                'kurtosis' => -1.1330702741786107,
                'range' => 9.0,
                'min' => -5.0,
                '25%' => -1.375,
                'median' => 0.8,
                '75%' => 2.825,
                'max' => 4.0,
            ],
        ];

        $this->assertInstanceOf(Report::class, $results);
        $this->assertEquals($expected, $results->toArray());
    }

    /**
     * @test
     */
    public function deduplicate() : void
    {
        $dataset = $this->dataset->deduplicate();

        $this->assertCount(6, $dataset);
    }

    /**
     * @test
     */
    public function testCount() : void
    {
        $this->assertEquals(6, $this->dataset->count());
        $this->assertCount(6, $this->dataset);
    }

    /**
     * @test
     */
    public function arrayAccess() : void
    {
        $expected = ['mean', 'furry', 'loner', -1.5];

        $this->assertEquals($expected, $this->dataset[1]);
    }

    /**
     * @test
     */
    public function iterate() : void
    {
        $expected = [
            ['nice', 'furry', 'friendly', 4.0],
            ['mean', 'furry', 'loner', -1.5],
            ['nice', 'rough', 'friendly', 2.6],
            ['mean', 'rough', 'friendly', -1.0],
            ['nice', 'rough', 'friendly', 2.9],
            ['nice', 'furry', 'loner', -5.0],
        ];

        $this->assertEquals($expected, iterator_to_array($this->dataset));
    }
}
