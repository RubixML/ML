<?php

use Rubix\Engine\Dataset;
use PHPUnit\Framework\TestCase;

class DatasetTest extends TestCase
{
    protected $data;

    public function setUp()
    {
        $data = [
            ['nice', 'furry', 'friendly', 'not monster'],
            ['mean', 'furry', 'loner', 'monster'],
            ['nice', 'rough', 'friendly', 'not monster'],
            ['mean', 'rough', 'friendly', 'monster'],
        ];

        $this->dataset = new Dataset($data);
    }

    public function test_build_dataset()
    {
        $this->assertInstanceOf(Dataset::class, $this->dataset);
    }
}
