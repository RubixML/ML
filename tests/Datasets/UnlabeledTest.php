<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Datasets;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\Datasets\Unlabeled;
use PHPUnit\Framework\TestCase;

use function Rubix\ML\array_transpose;

#[Group('Datasets')]
#[CoversClass(Unlabeled::class)]
class UnlabeledTest extends TestCase
{
    protected const array SAMPLES = [
        ['nice', 'furry', 'friendly', 4.0],
        ['mean', 'furry', 'loner', -1.5],
        ['nice', 'rough', 'friendly', 2.6],
        ['mean', 'rough', 'friendly', -1.0],
        ['nice', 'rough', 'friendly', 2.9],
        ['nice', 'furry', 'loner', -5.0],
    ];

    protected const array TYPES = [
        DataType::CATEGORICAL,
        DataType::CATEGORICAL,
        DataType::CATEGORICAL,
        DataType::CONTINUOUS,
    ];

    protected const array WEIGHTS = [
        1, 1, 2, 1, 2, 3,
    ];

    protected const int RANDOM_SEED = 0;

    protected Unlabeled $dataset;

    protected function setUp() : void
    {
        $this->dataset = new Unlabeled(samples: self::SAMPLES, verify: false);

        srand(self::RANDOM_SEED);
    }

    #[Test]
    public function stack() : void
    {
        $dataset1 = new Unlabeled(samples: [['sample1']]);
        $dataset2 = new Unlabeled(samples: [['sample2']]);
        $dataset3 = new Unlabeled(samples: [['sample3']]);

        $dataset = Unlabeled::stack([$dataset1, $dataset2, $dataset3]);

        $this->assertSame(3, $dataset->numSamples());
        $this->assertSame(1, $dataset->numFeatures());
    }

    #[Test]
    public function samples() : void
    {
        $this->assertSame(self::SAMPLES, $this->dataset->samples());
    }

    #[Test]
    public function sample() : void
    {
        $this->assertSame(self::SAMPLES[2], $this->dataset->sample(2));
        $this->assertSame(self::SAMPLES[5], $this->dataset->sample(5));
    }

    #[Test]
    public function numSamples() : void
    {
        $this->assertSame(6, $this->dataset->numSamples());
    }

    #[Test]
    public function feature() : void
    {
        $expected = array_column(self::SAMPLES, 2);

        $this->assertSame($expected, $this->dataset->feature(2));
    }

    #[Test]
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

        $this->assertSame($expected, $this->dataset->samples());
    }

    #[Test]
    public function numFeatures() : void
    {
        $this->assertSame(4, $this->dataset->numFeatures());
    }

    #[Test]
    public function featureType() : void
    {
        $this->assertEquals(DataType::categorical(), $this->dataset->featureType(0));
        $this->assertEquals(DataType::categorical(), $this->dataset->featureType(1));
        $this->assertEquals(DataType::categorical(), $this->dataset->featureType(2));
        $this->assertEquals(DataType::continuous(), $this->dataset->featureType(3));
    }

    #[Test]
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

    #[Test]
    public function uniqueTypes() : void
    {
        $this->assertCount(2, $this->dataset->uniqueTypes());
    }

    #[Test]
    public function homogeneous() : void
    {
        $this->assertFalse($this->dataset->homogeneous());
    }

    #[Test]
    public function shape() : void
    {
        $this->assertSame([6, 4], $this->dataset->shape());
    }

    public function testSize() : void
    {
        $this->assertSame(24, $this->dataset->size());
    }

    #[Test]
    public function features() : void
    {
        $expected = array_transpose(self::SAMPLES);

        $this->assertSame($expected, $this->dataset->features());
    }

    #[Test]
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

    #[Test]
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

        $this->assertSame($expected, $filtered->samples());
    }

    #[Test]
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

        $this->assertSame($expected, $dataset->samples());
    }

    #[Test]
    public function featuresByType() : void
    {
        $expected = array_slice(array_transpose(self::SAMPLES), 0, 3);

        $columns = $this->dataset->featuresByType(DataType::categorical());

        $this->assertSame($expected, $columns);
    }

    #[Test]
    public function empty() : void
    {
        $this->assertFalse($this->dataset->empty());
    }

    #[Test]
    public function randomize() : void
    {
        $samples = $this->dataset->samples();

        $this->dataset->randomize();

        $this->assertNotEquals($samples, $this->dataset->samples());
    }

    #[Test]
    public function head() : void
    {
        $subset = $this->dataset->head(3);

        $this->assertCount(3, $subset);
    }

    #[Test]
    public function tail() : void
    {
        $subset = $this->dataset->tail(3);

        $this->assertCount(3, $subset);
    }

    #[Test]
    public function take() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->take(3);

        $this->assertCount(3, $subset);
        $this->assertCount(3, $this->dataset);
    }

    #[Test]
    public function leave() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->leave();

        $this->assertCount(5, $subset);
        $this->assertCount(1, $this->dataset);
    }

    #[Test]
    public function slice() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->slice(2, 2);

        $this->assertCount(2, $subset);
        $this->assertCount(6, $this->dataset);
    }

    #[Test]
    public function splice() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->splice(offset: 2, n: 2);

        $this->assertCount(2, $subset);
        $this->assertCount(4, $this->dataset);
    }

    #[Test]
    public function split() : void
    {
        [$left, $right] = $this->dataset->split();

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    #[Test]
    public function fold() : void
    {
        $folds = $this->dataset->fold(2);

        $this->assertCount(2, $folds);
        $this->assertCount(3, $folds[0]);
        $this->assertCount(3, $folds[1]);
    }

    #[Test]
    public function batch() : void
    {
        $batches = $this->dataset->batch(2);

        $this->assertCount(3, $batches);
        $this->assertCount(2, $batches[0]);
        $this->assertCount(2, $batches[1]);
        $this->assertCount(2, $batches[2]);
    }

    #[Test]
    public function partition() : void
    {
        [$left, $right] = $this->dataset->splitByFeature(column: 2, value: 'loner');

        $this->assertCount(2, $left);
        $this->assertCount(4, $right);
    }

    #[Test]
    public function randomSubset() : void
    {
        $subset = $this->dataset->randomSubset(3);

        $this->assertCount(3, array_unique($subset->samples(), SORT_REGULAR));
    }

    #[Test]
    public function randomSubsetWithReplacement() : void
    {
        $subset = $this->dataset->randomSubsetWithReplacement(3);

        $this->assertCount(3, $subset);
    }

    #[Test]
    public function randomWeightedSubsetWithReplacement() : void
    {
        $subset = $this->dataset->randomWeightedSubsetWithReplacement(n: 3, weights: self::WEIGHTS);

        $this->assertCount(3, $subset);
    }

    #[Test]
    public function merge() : void
    {
        $this->assertCount(count(self::SAMPLES), $this->dataset);

        $dataset = new Unlabeled(samples: [['nice', 'furry', 'friendly', 4.7]]);

        $merged = $this->dataset->merge($dataset);

        $this->assertCount(count(self::SAMPLES) + 1, $merged);

        $this->assertSame(['nice', 'furry', 'friendly', 4.7], $merged->sample(6));
    }

    #[Test]
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

        $this->assertSame(count(current(self::SAMPLES)) + 1, $joined->numFeatures());

        $this->assertSame(['mean', 'furry', 'loner', -1.5, 2], $joined->sample(1));
        $this->assertSame(['nice', 'rough', 'friendly', 2.6, 3], $joined->sample(2));
    }

    #[Test]
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

        $this->assertEquals($expected, $results->toArray());
    }

    #[Test]
    public function deduplicate() : void
    {
        $dataset = $this->dataset->deduplicate();

        $this->assertCount(6, $dataset);
    }

    public function testCount() : void
    {
        $this->assertEquals(6, $this->dataset->count());
        $this->assertCount(6, $this->dataset);
    }

    #[Test]
    public function arrayAccess() : void
    {
        $expected = ['mean', 'furry', 'loner', -1.5];

        $this->assertSame($expected, $this->dataset[1]);
    }

    #[Test]
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

        $this->assertSame($expected, iterator_to_array($this->dataset));
    }
}
