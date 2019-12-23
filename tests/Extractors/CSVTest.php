<?php

namespace Rubix\ML\Tests\Extractors;

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\Extractor;
use PHPUnit\Framework\TestCase;
use IteratorAggregate;

class CSVTest extends TestCase
{
    /**
     * @var \Rubix\ML\Extractors\CSV;
     */
    protected $extractor;

    public function setUp() : void
    {
        $this->extractor = new CSV('tests/test.csv', ',', '');
    }

    public function test_build_extractor() : void
    {
        $this->assertInstanceOf(CSV::class, $this->extractor);
        $this->assertInstanceOf(Extractor::class, $this->extractor);
        $this->assertInstanceOf(IteratorAggregate::class, $this->extractor);
    }

    public function test_extract() : void
    {
        $records = $this->extractor->setHeader()->setOffset(1)->extract();

        $expected = [
            // ['attitude' => 'nice', 'appearance' => 'furry', 'sociability' => 'friendly', 'rating' => '4', 'class' => 'not monster'],
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
