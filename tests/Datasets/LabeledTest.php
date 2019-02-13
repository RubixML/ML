<?php

namespace Rubix\ML\Tests\Datasets;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Helpers\DataType;
use Rubix\ML\Datasets\DataFrame;
use PHPUnit\Framework\TestCase;
use ArrayIterator;

class LabeledTest extends TestCase
{
    protected $dataset;

    protected $samples;

    protected $labels;

    protected $types;

    protected $weights;

    public function setUp()
    {
        $this->samples = [
            ['nice', 'furry', 'friendly'],
            ['mean', 'furry', 'loner'],
            ['nice', 'rough', 'friendly'],
            ['mean', 'rough', 'friendly'],
            ['nice', 'rough', 'friendly'],
            ['nice', 'furry', 'loner'],
        ];

        $this->labels = [
            'not monster', 'monster', 'not monster',
            'monster', 'not monster', 'not monster',
        ];

        $this->types = [DataType::CATEGORICAL, DataType::CATEGORICAL, DataType::CATEGORICAL];

        $this->weights = [
            1, 1, 2, 1, 2, 3,
        ];

        $this->dataset = new Labeled($this->samples, $this->labels, false);
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
        $samples = new ArrayIterator($this->samples);
        $labels = new ArrayIterator($this->labels);

        $dataset = Labeled::fromIterator($samples, $labels);

        $this->assertInstanceOf(Labeled::class, $dataset);

        $this->assertEquals($this->samples, $dataset->samples());
        $this->assertEquals($this->labels, $dataset->labels());
    }

    public function test_get_labels()
    {
        $this->assertEquals($this->labels, $this->dataset->labels());
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
        $this->dataset->transformLabels(function ($label) {
            return $label === 'not monster' ? 0 : 1;
        });

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

    public function test_get_column_types()
    {
        $this->assertEquals($this->types, $this->dataset->types());
    }

    public function test_get_column_type()
    {
        $this->assertEquals($this->types[0], $this->dataset->columnType(0));
        $this->assertEquals($this->types[1], $this->dataset->columnType(1));
        $this->assertEquals($this->types[2], $this->dataset->columnType(2));
    }

    public function test_randomize()
    {
        $this->dataset->randomize();

        $this->assertTrue(true);
    }

    public function test_filter_by_column()
    {
        $filtered = $this->dataset->filterByColumn(2, function ($value) {
            return $value === 'friendly';
        });

        $outcome = [
            ['nice', 'furry', 'friendly'],
            ['nice', 'rough', 'friendly'],
            ['mean', 'rough', 'friendly'],
            ['nice', 'rough', 'friendly'],
        ];

        $labels = ['not monster', 'not monster', 'monster', 'not monster'];

        $this->assertEquals($outcome, $filtered->samples());
        $this->assertEquals($labels, $filtered->labels());
    }

    public function test_filter_by_label()
    {
        $filtered = $this->dataset->filterByLabel(function ($label) {
            return $label === 'not monster';
        });

        $outcome = [
            ['nice', 'furry', 'friendly'],
            ['nice', 'rough', 'friendly'],
            ['nice', 'rough', 'friendly'],
            ['nice', 'furry', 'loner'],
        ];

        $labels = ['not monster', 'not monster', 'not monster', 'not monster'];

        $this->assertEquals($outcome, $filtered->samples());
        $this->assertEquals($labels, $filtered->labels());
    }

    public function test_sort_by_column()
    {
        $this->dataset->sortByColumn(1);

        $sorted = array_column($this->samples, 1);

        array_multisort($sorted, $this->labels, SORT_ASC);

        $this->assertEquals($sorted, $this->dataset->column(1));
        $this->assertEquals($this->labels, $this->dataset->labels());
    }

    public function test_sort_by_label()
    {
        $this->dataset->sortByLabel();

        array_multisort($this->labels, $this->samples, SORT_ASC);

        $this->assertEquals($this->samples, $this->dataset->samples());
        $this->assertEquals($this->labels, $this->dataset->labels());
    }

    public function test_head()
    {
        $this->assertCount(3, $this->dataset->head(3));
    }

    public function test_tail()
    {
        $this->assertCount(3, $this->dataset->tail(3));
    }

    public function test_take_samples_from_dataset()
    {
        $this->assertCount(6, $this->dataset);

        $dataset = $this->dataset->take(3);

        $this->assertCount(3, $dataset);
        $this->assertCount(3, $this->dataset);
    }

    public function test_leave_samples_in_dataset()
    {
        $this->assertCount(6, $this->dataset);

        $dataset = $this->dataset->leave(1);

        $this->assertCount(5, $dataset);
        $this->assertCount(1, $this->dataset);
    }

    public function test_splice_dataset()
    {
        $this->assertCount(6, $this->dataset);

        $dataset = $this->dataset->splice(2, 2);

        $this->assertCount(2, $dataset);
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

    public function test_random_subset_with_replacement()
    {
        $subset = $this->dataset->randomSubsetWithReplacement(3);

        $this->assertCount(3, $subset);
    }

    public function test_random_weighted_subset_with_replacement()
    {
        $subset = $this->dataset->randomWeightedSubsetWithReplacement(3, $this->weights);

        $this->assertCount(3, $subset);
    }

    public function test_prepend_dataset()
    {
        $this->assertCount(count($this->samples), $this->dataset);

        $dataset = new Labeled([['nice', 'furry', 'friendly']], ['not monster']);

        $merged = $this->dataset->prepend($dataset);

        $this->assertCount(count($this->samples) + 1, $merged);

        $this->assertEquals(['nice', 'furry', 'friendly'], $merged->row(0));
        $this->assertEquals('not monster', $merged->label(6));
    }

    public function test_append_dataset()
    {
        $this->assertCount(count($this->samples), $this->dataset);

        $dataset = new Labeled([['nice', 'furry', 'friendly']], ['not monster']);

        $merged = $this->dataset->append($dataset);

        $this->assertCount(count($this->samples) + 1, $merged);

        $this->assertEquals(['nice', 'furry', 'friendly'], $merged->row(6));
        $this->assertEquals('not monster', $merged->label(6));
    }
}
