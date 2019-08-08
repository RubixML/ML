<?php

namespace Rubix\ML\Tests\Datasets;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Datasets\DataFrame;
use PHPUnit\Framework\TestCase;
use ArrayIterator;

class UnlabeledTest extends TestCase
{
    protected const SAMPLES = [
        ['nice', 'furry', 'friendly'],
        ['mean', 'furry', 'loner'],
        ['nice', 'rough', 'friendly'],
        ['mean', 'rough', 'friendly'],
        ['nice', 'rough', 'friendly'],
        ['nice', 'furry', 'loner'],
    ];

    protected const TYPES = [
        DataType::CATEGORICAL,
        DataType::CATEGORICAL,
        DataType::CATEGORICAL,
    ];

    protected const WEIGHTS = [
        1, 1, 2, 1, 2, 3,
    ];

    protected const RANDOM_SEED = 0;

    protected $dataset;

    public function setUp()
    {
        $this->dataset = new Unlabeled(self::SAMPLES, false);

        srand(self::RANDOM_SEED);
    }

    public function test_build_dataset()
    {
        $this->assertInstanceOf(Unlabeled::class, $this->dataset);
        $this->assertInstanceOf(DataFrame::class, $this->dataset);
        $this->assertInstanceOf(Dataset::class, $this->dataset);
    }

    public function test_stack_datasets()
    {
        $dataset1 = new Unlabeled([['sample1']]);
        $dataset2 = new Unlabeled([['sample2']]);
        $dataset3 = new Unlabeled([['sample3']]);

        $dataset = Unlabeled::stack([$dataset1, $dataset2, $dataset3]);

        $this->assertInstanceOf(Unlabeled::class, $dataset);

        $this->assertEquals(3, $dataset->numRows());
        $this->assertEquals(1, $dataset->numColumns());
    }

    public function test_from_iterator()
    {
        $samples = new ArrayIterator(self::SAMPLES);

        $dataset = Unlabeled::fromIterator($samples);

        $this->assertInstanceOf(Unlabeled::class, $dataset);

        $this->assertEquals(self::SAMPLES, $dataset->samples());
    }

    public function test_randomize()
    {
        $samples = $this->dataset->samples();

        $this->dataset->randomize();

        $this->assertNotEquals($samples, $this->dataset->samples());
    }

    public function test_filter_by_column()
    {
        $isFriendly = function ($value) {
            return $value === 'friendly';
        };

        $filtered = $this->dataset->filterByColumn(2, $isFriendly);

        $outcome = [
            ['nice', 'furry', 'friendly'],
            ['nice', 'rough', 'friendly'],
            ['mean', 'rough', 'friendly'],
            ['nice', 'rough', 'friendly'],
        ];

        $this->assertEquals($outcome, $filtered->samples());
    }

    public function test_sort_by_column()
    {
        $this->dataset->sortByColumn(2);

        $sorted = array_column(self::SAMPLES, 2);

        sort($sorted);

        $this->assertEquals($sorted, $this->dataset->column(2));
    }

    public function test_head()
    {
        $subset = $this->dataset->head(3);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    public function test_tail()
    {
        $subset = $this->dataset->tail(3);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    public function test_take()
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->take(3);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(3, $subset);
        $this->assertCount(3, $this->dataset);
    }

    public function test_leave()
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->leave(1);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(5, $subset);
        $this->assertCount(1, $this->dataset);
    }

    public function test_slice_dataset()
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->slice(2, 2);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(2, $subset);
        $this->assertCount(6, $this->dataset);
    }

    public function test_splice_dataset()
    {
        $this->assertCount(6, $this->dataset);

        $subset = $this->dataset->splice(2, 2);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(2, $subset);
        $this->assertCount(4, $this->dataset);
    }

    public function test_split_dataset()
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
        [$left, $right] = $this->dataset->partition(2, 'loner');

        $this->assertInstanceOf(Unlabeled::class, $left);
        $this->assertInstanceOf(Unlabeled::class, $right);

        $this->assertCount(2, $left);
        $this->assertCount(4, $right);
    }

    public function test_random_subset()
    {
        $subset = $this->dataset->randomSubset(3);

        $this->assertCount(3, array_unique($subset->samples(), SORT_REGULAR));
    }

    public function test_random_subset_with_replacement()
    {
        $subset = $this->dataset->randomSubsetWithReplacement(3);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    public function test_random_weighted_subset_with_replacement()
    {
        $subset = $this->dataset->randomWeightedSubsetWithReplacement(3, self::WEIGHTS);

        $this->assertInstanceOf(Unlabeled::class, $subset);
        $this->assertCount(3, $subset);
    }

    public function test_prepend_dataset()
    {
        $this->assertCount(count(self::SAMPLES), $this->dataset);

        $dataset = new Unlabeled([['nice', 'furry', 'friendly']]);

        $merged = $this->dataset->prepend($dataset);

        $this->assertCount(count(self::SAMPLES) + 1, $merged);

        $this->assertEquals(['nice', 'furry', 'friendly'], $merged->row(0));
    }

    public function test_append_dataset()
    {
        $this->assertCount(count(self::SAMPLES), $this->dataset);

        $dataset = new Unlabeled([['nice', 'furry', 'friendly']]);

        $merged = $this->dataset->append($dataset);

        $this->assertCount(count(self::SAMPLES) + 1, $merged);

        $this->assertEquals(['nice', 'furry', 'friendly'], $merged->row(6));
    }
}
