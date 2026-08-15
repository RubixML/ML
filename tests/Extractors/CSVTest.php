<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Extractors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Extractors\CSV;
use PHPUnit\Framework\TestCase;

#[Group('Extractors')]
#[CoversClass(CSV::class)]
class CSVTest extends TestCase
{
    protected CSV $extractor;

    protected function setUp() : void
    {
        $this->extractor = new CSV(
            path: 'tests/test.csv',
            header: true,
            delimiter: ',',
            enclosure: '"'
        );
    }

    public function testHeader() : void
    {
        $expected = [
            'attitude', 'texture', 'sociability', 'rating', 'class',
        ];

        $this->assertEquals($expected, $this->extractor->header());
    }

    public function testExtractExport() : void
    {
        $expected = [
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => '4', 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => '-1.5', 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '2.6', 'class' => 'not monster'],
            ['attitude' => 'mean', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '-1', 'class' => 'monster'],
            ['attitude' => 'nice', 'texture' => 'rough', 'sociability' => 'friendly', 'rating' => '2.9', 'class' => 'not monster'],
            ['attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'loner', 'rating' => '-5', 'class' => 'not monster'],
        ];

        $records = iterator_to_array($this->extractor, false);

        $this->assertEquals($expected, $records);

        $expected = [
            'attitude', 'texture', 'sociability', 'rating', 'class',
        ];

        $header = $this->extractor->header();

        $this->assertEquals($expected, $header);

        $this->extractor->export(iterator: $records, overwrite: true);

        $this->assertFileExists('tests/test.csv');
    }
}
