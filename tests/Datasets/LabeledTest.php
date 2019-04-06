<?php

namespace Rubix\ML\Tests\Datasets;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\DataFrame;
use Rubix\ML\Other\Helpers\DataType;
use PHPUnit\Framework\TestCase;
use ArrayIterator;

class LabeledTest extends TestCase
{
    protected const SAMPLES = [
        ['nice', 'furry', 'friendly'],
        ['mean', 'furry', 'loner'],
        ['nice', 'rough', 'friendly'],
        ['mean', 'rough', 'friendly'],
        ['nice', 'rough', 'friendly'],
        ['nice', 'furry', 'loner'],
    ];

    protected const LABELS = [
        'not monster', 'monster', 'not monster',
        'monster', 'not monster', 'not monster',
    ];

    protected const TYPES = [
        DataType::CATEGORICAL,
        DataType::CATEGORICAL,
        DataType::CATEGORICAL,
    ];

    protected const WEIGHTS = [
        1, 1, 2, 1, 2, 3,
    ];

    protected const RANDOM_SEED = 1;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = new Labeled(self::SAMPLES, self::LABELS, false);

        srand(self::RANDOM_SEED);
    }

    public function test_build_dataset()
    {
        $this->assertInstanceOf(Labeled::class, $this->dataset);
        $this->assertInstanceOf(DataFrame::class, $this->dataset);
        $this->assertInstanceOf(Dataset::class, $this->dataset);
    }

    public function test_stack_datasets()
    {
        $dataset1 = new Labeled([['sample1']], ['label1']);
        $dataset2 = new Labeled([['sample2']], ['label2']);
        $dataset3 = new Labeled([['sample3']], ['label3']);

        $dataset = Labeled::stack([$dataset1, $dataset2, $dataset3]);

        $this->assertInstanceOf(Labeled::class, $dataset);

        $this->assertEquals(3, $dataset->numRows());
        $this->assertEquals(1, $dataset->numColumns());
    }

    public function test_from_iterator()
    {
        $samples = new ArrayIterator(self::SAMPLES);
        $labels = new ArrayIterator(self::LABELS);

        $dataset = Labeled::fromIterator($samples, $labels);

        $this->assertInstanceOf(Labeled::class, $dataset);

        $this->assertEquals(self::SAMPLES, $dataset->samples());
        $this->assertEquals(self::LABELS, $dataset->labels());
    }

    public function test_get_labels()
    {
        $this->assertEquals(self::LABELS, $this->dataset->labels());
    }

    public function test_zip()
    {
        $outcome = [
            ['nice', 'furry', 'friendly', 'not monster'],
            ['mean', 'furry', 'loner', 'monster'],
            ['nice', 'rough', 'friendly', 'not monster'],
            ['mean', 'rough', 'friendly', 'monster'],
            ['nice', 'rough', 'friendly', 'not monster'],
            ['nice', 'furry', 'loner', 'not monster'],
        ];
        
        $this->assertEquals($outcome, $this->dataset->zip());
    }

    public function test_transform_labels()
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

    public function test_get_label()
    {
        $this->assertEquals('not monster', $this->dataset->label(0));
        $this->assertEquals('monster', $this->dataset->label(1));
    }

    public function test_label_type()
    {
        $this->assertEquals(DataType::CATEGORICAL, $this->dataset->labelType());
    }

    public function test_possible_outcomes()
    {
        $this->assertEquals(
            ['not monster', 'monster'],
            $this->dataset->possibleOutcomes()
        );
    }

    public function test_randomize()
    {
        $samples = $this->dataset->samples();
        $labels = $this->dataset->labels();

        $this->dataset->randomize();

        $this->assertNotEquals($samples, $this->dataset->samples());
        $this->assertNotEquals($labels, $this->dataset->labels());
    }

    public function test_filter_by_column()
    {
        $isFriendly = function ($value) {
            return $value === 'friendly';
        };

        $filtered = $this->dataset->filterByColumn(2, $isFriendly);

        $samples = [
            ['nice', 'furry', 'friendly'],
            ['nice', 'rough', 'friendly'],
            ['mean', 'rough', 'friendly'],
            ['nice', 'rough', 'friendly'],
        ];

        $labels = ['not monster', 'not monster', 'monster', 'not monster'];

        $this->assertEquals($samples, $filtered->samples());
        $this->assertEquals($labels, $filtered->labels());
    }

    public function test_filter_by_label()
    {
        $notMonster = function ($label) {
            return $label === 'not monster';
        };

        $filtered = $this->dataset->filterByLabel($notMonster);

        $samples = [
            ['nice', 'furry', 'friendly'],
            ['nice', 'rough', 'friendly'],
            ['nice', 'rough', 'friendly'],
            ['nice', 'furry', 'loner'],
        ];

        $labels = ['not monster', 'not monster', 'not monster', 'not monster'];

        $this->assertEquals($samples, $filtered->samples());
        $this->assertEquals($labels, $filtered->labels());
    }

    public function test_sort_by_column()
    {
        $this->dataset->sortByColumn(1);

        $sorted = array_column(self::SAMPLES, 1);

        $labels = self::LABELS;

        array_multisort($sorted, $labels, SORT_ASC);

        $this->assertEquals($sorted, $this->dataset->column(1));
        $this->assertEquals($labels, $this->dataset->labels());
    }

    public function test_sort_by_label()
    {
        $this->dataset->sortByLabel();

        $samples = self::SAMPLES;
        $labels = self::LABELS;

        array_multisort($labels, $samples, SORT_ASC);

        $this->assertEquals($samples, $this->dataset->samples());
        $this->assertEquals($labels, $this->dataset->labels());
    }

    public function test_head()
    {
        $subset = $this->dataset->head(3);

        $this->assertInstanceOf(Labeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    public function test_tail()
    {
        $subset = $this->dataset->tail(3);

        $this->assertInstanceOf(Labeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    public function test_take()
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->take(3);

        $this->assertCount(3, $subset);
        $this->assertCount(3, $this->dataset);
    }

    public function test_leave()
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->leave(1);

        $this->assertCount(5, $subset);
        $this->assertCount(1, $this->dataset);
    }

    public function test_splice_dataset()
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->splice(2, 2);

        $this->assertCount(2, $subset);
        $this->assertCount(4, $this->dataset);
    }

    public function test_split_dataset()
    {
        [$left, $right] = $this->dataset->split(0.5);

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    public function test_stratified_split()
    {
        [$left, $right] = $this->dataset->split(0.5);

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    public function test_fold_dataset()
    {
        $folds = $this->dataset->fold(2);

        $this->assertCount(2, $folds);
        $this->assertCount(3, $folds[0]);
        $this->assertCount(3, $folds[1]);
    }

    public function test_stratified_fold()
    {
        $folds = $this->dataset->stratifiedFold(2);

        $this->assertCount(2, $folds);
        $this->assertCount(3, $folds[0]);
        $this->assertCount(3, $folds[1]);
    }

    public function test_stratify()
    {
        $strata = $this->dataset->stratify();

        $this->assertCount(2, $strata['monster']);
        $this->assertCount(4, $strata['not monster']);
    }

    public function test_batch_dataset()
    {
        $batches = $this->dataset->batch(2);

        $this->assertCount(3, $batches);
        $this->assertCount(2, $batches[0]);
        $this->assertCount(2, $batches[1]);
        $this->assertCount(2, $batches[2]);
    }

    public function test_partition()
    {
        [$left, $right] = $this->dataset->partition(1, 'rough');

        $this->assertInstanceOf(Labeled::class, $left);
        $this->assertInstanceOf(Labeled::class, $right);

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    public function test_random_subset_with_replacement()
    {
        $subset = $this->dataset->randomSubsetWithReplacement(3);

        $this->assertCount(3, $subset);
    }

    public function test_random_weighted_subset_with_replacement()
    {
        $subset = $this->dataset->randomWeightedSubsetWithReplacement(3, self::WEIGHTS);

        $this->assertCount(3, $subset);
    }

    public function test_prepend_dataset()
    {
        $this->assertCount(count(self::SAMPLES), $this->dataset);

        $dataset = new Labeled([['nice', 'furry', 'friendly']], ['not monster']);

        $merged = $this->dataset->prepend($dataset);

        $this->assertCount(count(self::SAMPLES) + 1, $merged);

        $this->assertEquals(['nice', 'furry', 'friendly'], $merged->row(0));
        $this->assertEquals('not monster', $merged->label(6));
    }

    public function test_append_dataset()
    {
        $this->assertCount(count(self::SAMPLES), $this->dataset);

        $dataset = new Labeled([['nice', 'furry', 'friendly']], ['not monster']);

        $merged = $this->dataset->append($dataset);

        $this->assertCount(count(self::SAMPLES) + 1, $merged);

        $this->assertEquals(['nice', 'furry', 'friendly'], $merged->row(6));
        $this->assertEquals('not monster', $merged->label(6));
    }
}
