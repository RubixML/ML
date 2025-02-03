<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Datasets;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Report;
use Rubix\ML\DataType;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Datasets\Unlabeled;
use PHPUnit\Framework\TestCase;

use function Rubix\ML\array_transpose;

#[Group('Datasets')]
#[CoversClass(Labeled::class)]
class LabeledTest extends TestCase
{
    protected const array SAMPLES = [
        ['nice', 'furry', 'friendly', 4.0],
        ['mean', 'furry', 'loner', -1.5],
        ['nice', 'rough', 'friendly', 2.6],
        ['mean', 'rough', 'friendly', -1.0],
        ['nice', 'rough', 'friendly', 2.9],
        ['nice', 'furry', 'loner', -5.0],
    ];

    protected const array LABELS = [
        'not monster', 'monster', 'not monster',
        'monster', 'not monster', 'not monster',
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

    protected const int RANDOM_SEED = 1;

    protected Labeled $dataset;

    protected function setUp() : void
    {
        ini_set('precision', '14');

        $this->dataset = new Labeled(
            samples: self::SAMPLES,
            labels: self::LABELS,
            verify: false
        );

        srand(self::RANDOM_SEED);
    }

    public function testFromIterator() : void
    {
        $dataset = Labeled::fromIterator(new NDJSON('tests/test.ndjson'));

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertEquals(self::SAMPLES, $dataset->samples());
        $this->assertEquals(self::LABELS, $dataset->labels());
    }

    public function testStack() : void
    {
        $dataset1 = new Labeled(samples: [['sample1']], labels: ['label1']);
        $dataset2 = new Labeled(samples: [['sample2']], labels: ['label2']);
        $dataset3 = new Labeled(samples: [['sample3']], labels: ['label3']);

        $dataset = Labeled::stack([$dataset1, $dataset2, $dataset3]);

        $this->assertInstanceOf(Labeled::class, $dataset);

        $this->assertEquals(3, $dataset->numSamples());
        $this->assertEquals(1, $dataset->numFeatures());
    }

    public function testExamples() : void
    {
        $this->assertEquals(self::SAMPLES, $this->dataset->samples());
    }

    public function testSample() : void
    {
        $this->assertEquals(self::SAMPLES[2], $this->dataset->sample(2));
        $this->assertEquals(self::SAMPLES[5], $this->dataset->sample(5));
    }

    public function testNumSamples() : void
    {
        $this->assertEquals(6, $this->dataset->numSamples());
    }

    public function testFeature() : void
    {
        $expected = array_column(self::SAMPLES, 2);

        $this->assertEquals($expected, $this->dataset->feature(2));
    }

    public function testDropFeature() : void
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

    public function testNumFeatures() : void
    {
        $this->assertEquals(4, $this->dataset->numFeatures());
    }

    public function testFeatureType() : void
    {
        $this->assertEquals(DataType::categorical(), $this->dataset->featureType(0));
        $this->assertEquals(DataType::categorical(), $this->dataset->featureType(1));
        $this->assertEquals(DataType::categorical(), $this->dataset->featureType(2));
        $this->assertEquals(DataType::continuous(), $this->dataset->featureType(3));
    }

    public function testFeatureTypes() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::categorical(),
            DataType::categorical(),
            DataType::continuous(),
        ];

        $this->assertEquals($expected, $this->dataset->featureTypes());
    }

    public function testUniqueTypes() : void
    {
        $this->assertCount(2, $this->dataset->uniqueTypes());
    }

    public function testHomogeneous() : void
    {
        $this->assertFalse($this->dataset->homogeneous());
    }

    public function testShape() : void
    {
        $this->assertEquals([6, 4], $this->dataset->shape());
    }

    public function testSize() : void
    {
        $this->assertEquals(24, $this->dataset->size());
    }

    public function testFeatures() : void
    {
        $expected = array_transpose(self::SAMPLES);

        $this->assertEquals($expected, $this->dataset->features());
    }

    public function testTypes() : void
    {
        $expected = [
            DataType::categorical(),
            DataType::categorical(),
            DataType::categorical(),
            DataType::continuous(),
            DataType::categorical(),
        ];

        $this->assertEquals($expected, $this->dataset->types());
    }

    public function testFeaturesByType() : void
    {
        $expected = array_slice(array_transpose(self::SAMPLES), 0, 3);

        $columns = $this->dataset->featuresByType(DataType::categorical());

        $this->assertEquals($expected, $columns);
    }

    public function testEmpty() : void
    {
        $this->assertFalse($this->dataset->empty());
    }

    public function testLabels() : void
    {
        $this->assertEquals(self::LABELS, $this->dataset->labels());
    }

    public function testTransformLabels() : void
    {
        $transformer = function ($label) {
            return $label === 'not monster' ? 0 : 1;
        };

        $this->dataset->transformLabels($transformer);

        $expected = [
            0, 1, 0, 1, 0, 0,
        ];

        $this->assertEquals($expected, $this->dataset->labels());
    }

    public function testLabel() : void
    {
        $this->assertEquals('not monster', $this->dataset->label(0));
        $this->assertEquals('monster', $this->dataset->label(1));
    }

    public function testLabelType() : void
    {
        $this->assertEquals(DataType::categorical(), $this->dataset->labelType());
    }

    public function testPossibleOutcomes() : void
    {
        $this->assertEquals(
            ['not monster', 'monster'],
            $this->dataset->possibleOutcomes()
        );
    }

    public function testRandomize() : void
    {
        $samples = $this->dataset->samples();
        $labels = $this->dataset->labels();

        $this->dataset->randomize();

        $this->assertNotEquals($samples, $this->dataset->samples());
        $this->assertNotEquals($labels, $this->dataset->labels());
    }

    public function testFilter() : void
    {
        $isFriendly = function ($record) {
            return $record[2] === 'friendly';
        };

        $filtered = $this->dataset->filter($isFriendly);

        $samples = [
            ['nice', 'furry', 'friendly', 4.0],
            ['nice', 'rough', 'friendly', 2.6],
            ['mean', 'rough', 'friendly', -1.0],
            ['nice', 'rough', 'friendly', 2.9],
        ];

        $labels = ['not monster', 'not monster', 'monster', 'not monster'];

        $this->assertEquals($samples, $filtered->samples());
        $this->assertEquals($labels, $filtered->labels());
    }

    public function testHead() : void
    {
        $subset = $this->dataset->head(3);

        $this->assertInstanceOf(Labeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    public function testTail() : void
    {
        $subset = $this->dataset->tail(3);

        $this->assertInstanceOf(Labeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    public function testTake() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->take(3);

        $this->assertCount(3, $subset);
        $this->assertCount(3, $this->dataset);
    }

    public function testLeave() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->leave(1);

        $this->assertCount(5, $subset);
        $this->assertCount(1, $this->dataset);
    }

    public function testSlice() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->slice(2, 2);

        $this->assertInstanceOf(Labeled::class, $subset);
        $this->assertCount(2, $subset);
        $this->assertCount(6, $this->dataset);
    }

    public function testSplice() : void
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->splice(2, 2);

        $this->assertInstanceOf(Labeled::class, $subset);
        $this->assertCount(2, $subset);
        $this->assertCount(4, $this->dataset);
    }

    public function testSplit() : void
    {
        [$left, $right] = $this->dataset->split();

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    public function testStratifiedSplit() : void
    {
        [$left, $right] = $this->dataset->stratifiedSplit(0.5);

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    public function testFold() : void
    {
        $folds = $this->dataset->fold(2);

        $this->assertCount(2, $folds);
        $this->assertCount(3, $folds[0]);
        $this->assertCount(3, $folds[1]);
    }

    public function testStratifiedFold() : void
    {
        $folds = $this->dataset->stratifiedFold(2);

        $this->assertCount(2, $folds);
        $this->assertCount(3, $folds[0]);
        $this->assertCount(3, $folds[1]);
    }

    public function testStratifyByLabel() : void
    {
        $strata = $this->dataset->stratifyByLabel();

        $this->assertCount(2, $strata['monster']);
        $this->assertCount(4, $strata['not monster']);
    }

    public function testBatch() : void
    {
        $batches = $this->dataset->batch(2);

        $this->assertCount(3, $batches);
        $this->assertCount(2, $batches[0]);
        $this->assertCount(2, $batches[1]);
        $this->assertCount(2, $batches[2]);
    }

    public function testPartition() : void
    {
        [$left, $right] = $this->dataset->splitByFeature(1, 'rough');

        $this->assertInstanceOf(Labeled::class, $left);
        $this->assertInstanceOf(Labeled::class, $right);

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    public function testRandomSubset() : void
    {
        $subset = $this->dataset->randomSubset(3);

        $this->assertCount(3, array_unique($subset->samples(), SORT_REGULAR));
    }

    public function testRandomSubsetWithReplacement() : void
    {
        $subset = $this->dataset->randomSubsetWithReplacement(3);

        $this->assertCount(3, $subset);
    }

    public function testRandomWeightedSubsetWithReplacement() : void
    {
        $subset = $this->dataset->randomWeightedSubsetWithReplacement(3, self::WEIGHTS);

        $this->assertCount(3, $subset);
    }

    public function testMerge() : void
    {
        $this->assertCount(count(self::SAMPLES), $this->dataset);

        $dataset = new Labeled([['nice', 'furry', 'friendly', 4.7]], ['not monster']);

        $merged = $this->dataset->merge($dataset);

        $this->assertCount(count(self::SAMPLES) + 1, $merged);

        $this->assertEquals(['nice', 'furry', 'friendly', 4.7], $merged->sample(6));
        $this->assertEquals('not monster', $merged->label(6));
    }

    public function testJoin() : void
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
        $this->assertEquals(self::LABELS, $joined->labels());
    }

    public function testSort() : void
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

    public function testDescribe() : void
    {
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
            [
                'offset' => 4,
                'type' => 'categorical',
                'num categories' => 2,
                'probabilities' => [
                    'not monster' => 0.6666666666666666,
                    'monster' => 0.3333333333333333,
                ],
            ],
        ];

        $results = $this->dataset->describe();

        $this->assertInstanceOf(Report::class, $results);
        $this->assertEquals($expected, $results->toArray());
    }

    public function testDescribeByLabel() : void
    {
        $expected = [
            'not monster' => [
                [
                    'offset' => 0,
                    'type' => 'categorical',
                    'num categories' => 1,
                    'probabilities' => [
                        'nice' => 1,
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
                        'friendly' => 0.75,
                        'loner' => 0.25,
                    ],
                ],
                [
                    'offset' => 3,
                    'type' => 'continuous',
                    'mean' => 1.125,
                    'variance' => 12.776875,
                    'standard deviation' => 3.574475485997911,
                    'skewness' => -1.0795676577113944,
                    'kurtosis' => -0.7175867765792474,
                    'range' => 9.0,
                    'min' => -5.0,
                    '25%' => 0.6999999999999993,
                    'median' => 2.75,
                    '75%' => 3.175,
                    'max' => 4.0,
                ],
                [
                    'offset' => 4,
                    'type' => 'categorical',
                    'num categories' => 1,
                    'probabilities' => [
                        'not monster' => 1.0,
                    ],
                ],
            ],
            'monster' => [
                [
                    'offset' => 0,
                    'type' => 'categorical',
                    'num categories' => 1,
                    'probabilities' => [
                        'mean' => 1,
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
                        'friendly' => 0.5,
                        'loner' => 0.5,
                    ],
                ],
                [
                    'offset' => 3,
                    'type' => 'continuous',
                    'mean' => -1.25,
                    'variance' => 0.0625,
                    'standard deviation' => 0.25,
                    'skewness' => 0.0,
                    'kurtosis' => -2.0,
                    'range' => 0.5,
                    'min' => -1.5,
                    '25%' => -1.375,
                    'median' => -1.25,
                    '75%' => -1.125,
                    'max' => -1.0,
                ],
                [
                    'offset' => 4,
                    'type' => 'categorical',
                    'num categories' => 1,
                    'probabilities' => [
                        'monster' => 1.0,
                    ],
                ],
            ],
        ];

        $results = $this->dataset->describeByLabel();

        $this->assertInstanceOf(Report::class, $results);
        $this->assertEquals($expected, $results->toArray());
    }

    public function testDeduplicate() : void
    {
        $dataset = $this->dataset->deduplicate();

        $this->assertCount(6, $dataset);
    }

    public function testTestCount() : void
    {
        $this->assertEquals(6, $this->dataset->count());
        $this->assertCount(6, $this->dataset);
    }

    public function testArrayAccess() : void
    {
        $expected = ['mean', 'furry', 'loner', -1.5, 'monster'];

        $this->assertEquals($expected, $this->dataset[1]);
    }

    public function testIterate() : void
    {
        $expected = [
            ['nice', 'furry', 'friendly', 4.0, 'not monster'],
            ['mean', 'furry', 'loner', -1.5, 'monster'],
            ['nice', 'rough', 'friendly', 2.6, 'not monster'],
            ['mean', 'rough', 'friendly', -1.0, 'monster'],
            ['nice', 'rough', 'friendly', 2.9, 'not monster'],
            ['nice', 'furry', 'loner', -5.0, 'not monster'],
        ];

        $this->assertEquals($expected, iterator_to_array($this->dataset));
    }
}
