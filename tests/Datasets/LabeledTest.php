<?php

namespace Rubix\Tests\Datasets;

use Rubix\ML\Datasets\Dataset;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Structures\DataFrame;
use PHPUnit\Framework\TestCase;

class LabeledTest extends TestCase
{
    protected $dataset;

    protected $samples;

    protected $labels;

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

        $this->dataset = new Labeled($this->samples, $this->labels);
    }

    public function test_build_dataset()
    {
        $this->assertInstanceOf(Labeled::class, $this->dataset);
        $this->assertInstanceOf(DataFrame::class, $this->dataset);
        $this->assertInstanceOf(Dataset::class, $this->dataset);
    }

    public function test_get_labels()
    {
        $this->assertEquals($this->labels, $this->dataset->labels());
    }

    public function test_get_label()
    {
        $this->assertEquals('not monster', $this->dataset->label(0));
        $this->assertEquals('monster', $this->dataset->label(1));
    }

    public function test_possible_outcomes()
    {
        $this->assertEquals(['not monster', 'monster'],
            $this->dataset->possibleOutcomes());
    }

    public function test_randomize()
    {
        $this->dataset->randomize();

        $this->assertTrue(true);
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
        list($left, $right) = $this->dataset->split(0.5);

        $this->assertCount(3, $left);
        $this->assertCount(3, $right);
    }

    public function test_stratified_split()
    {
        list($left, $right) = $this->dataset->split(0.5);

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

    public function test_save_and_restore()
    {
        $this->assertFalse(file_exists(__DIR__ . '/test.dataset'));

        $this->dataset->save(__DIR__ . '/test.dataset');

        $this->assertFileExists(__DIR__ . '/test.dataset');

        $dataset = Labeled::restore(__DIR__ . '/test.dataset');

        $this->assertInstanceOf(Labeled::class, $dataset);
        $this->assertInstanceOf(Dataset::class, $dataset);

        $this->assertTrue(unlink(__DIR__ . '/test.dataset'));
    }

    public function test_prepend_dataset()
    {
        $this->assertCount(count($this->samples), $this->dataset);

        $dataset = new Labeled([['nice', 'furry', 'friendly']], ['not monster']);

        $this->dataset->prepend($dataset);

        $this->assertCount(count($this->samples) + 1, $this->dataset);

        $this->assertEquals(['nice', 'furry', 'friendly'], $this->dataset->row(0));
        $this->assertEquals('not monster', $this->dataset->label(0));
    }

    public function test_append_dataset()
    {
        $this->assertCount(count($this->samples), $this->dataset);

        $dataset = new Labeled([['nice', 'furry', 'friendly']], ['not monster']);

        $this->dataset->append($dataset);

        $this->assertCount(count($this->samples) + 1, $this->dataset);

        $row = count($this->dataset) - 1;

        $this->assertEquals(['nice', 'furry', 'friendly'], $this->dataset->row($row));
        $this->assertEquals('not monster', $this->dataset->label($row));
    }
}
