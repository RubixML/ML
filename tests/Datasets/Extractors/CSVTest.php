<?php

namespace Rubix\ML\Tests\Datasets\Extractors;

use Rubix\ML\Datasets\Extractors\CSV;
use Rubix\ML\Datasets\Extractors\Extractor;
use PHPUnit\Framework\TestCase;

class CSVTest extends TestCase
{
    /**
     * @var \Rubix\ML\Datasets\Extractors\Extractor;
     */
    protected $extractor;

    public function setUp() : void
    {
        $this->extractor = new CSV('tests/test.csv', ',', true, null);
    }

    public function test_build_extractor() : void
    {
        $this->assertInstanceOf(CSV::class, $this->extractor);
        $this->assertInstanceOf(Extractor::class, $this->extractor);
    }

    public function test_extract() : void
    {
        $records = $this->extractor->extract();

        $expected = [
            ['attitude' => 'nice', 'appearance' => 'furry', 'sociability' => 'friendly', 'rating' => '4', 'class' => 'not monster'],
            ['attitude' => 'mean', 'appearance' => 'furry', 'sociability' => 'loner', 'rating' => '-1.5', 'class' => 'monster'],
            ['attitude' => 'nice', 'appearance' => 'rough', 'sociability' => 'friendly', 'rating' => '2.6', 'class' => 'not monster'],
            ['attitude' => 'mean', 'appearance' => 'rough', 'sociability' => 'friendly', 'rating' => '-1', 'class' => 'monster'],
            ['attitude' => 'nice', 'appearance' => 'rough', 'sociability' => 'friendly', 'rating' => '2.9', 'class' => 'not monster'],
            ['attitude' => 'nice', 'appearance' => 'furry', 'sociability' => 'loner', 'rating' => '-5', 'class' => 'not monster'],
        ];

        $records = is_array($records) ? $records : iterator_to_array($records);

        $this->assertEquals($expected, array_values($records));
    }
}
