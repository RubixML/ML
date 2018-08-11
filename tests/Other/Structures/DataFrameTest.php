<?php

namespace Rubix\Tests\Other\Structures;

use Rubix\ML\Other\Structures\DataFrame;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;

class DataFrameTest extends TestCase
{
    protected $dataframe;

    protected $samples;

    protected $stored;

    public function setUp()
    {
        $this->samples = [
            ['nice', 'furry', 'friendly'],
            ['mean', 'furry', 'loner'],
            ['nice', 'rough', 'friendly'],
            ['mean', null, 'friendly'],
            ['nice', 'rough', 'friendly'],
            ['nice', 'furry', 'loner'],
        ];

        $this->stored = [
            ['nice', 'furry', 'friendly'],
            ['mean', 'furry', 'loner'],
            ['nice', 'rough', 'friendly'],
            ['mean', '?', 'friendly'],
            ['nice', 'rough', 'friendly'],
            ['nice', 'furry', 'loner'],
        ];

        $this->dataframe = new DataFrame($this->samples, '?');
    }

    public function test_build_data_frame()
    {
        $this->assertInstanceOf(DataFrame::class, $this->dataframe);
    }

    public function test_get_samples()
    {
        $this->assertEquals($this->stored, $this->dataframe->samples());
    }

    public function test_get_row()
    {
        $this->assertEquals($this->stored[2], $this->dataframe->row(2));
        $this->assertEquals($this->stored[5], $this->dataframe->row(5));
    }

    public function test_num_rows()
    {
        $this->assertEquals(6, $this->dataframe->numRows());
    }

    public function test_get_column()
    {
        $this->assertEquals(array_column($this->stored, 2),
            $this->dataframe->column(2));
    }

    public function test_get_column_indices()
    {
        $this->assertEquals(array_keys($this->stored[0]),
            $this->dataframe->indices());
    }

    public function test_get_num_columns()
    {
        $this->assertEquals(3, $this->dataframe->numColumns());
    }

    public function test_build_data_frame_invalid_placeholder()
    {
        $this->expectException(InvalidArgumentException::class);

        $dataframe = new DataFrame($this->samples, ['bad']);
    }
}
