<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Extractors;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\ColumnPicker;
use PHPUnit\Framework\TestCase;

#[Group('Extractors')]
#[CoversClass(ColumnPicker::class)]
class ColumnPickerTest extends TestCase
{
    protected ColumnPicker $extractor;

    protected function setUp() : void
    {
        $this->extractor = new ColumnPicker(
            iterator: new CSV(path: 'tests/test.csv', header: true),
            columns: [
                'attitude', 'texture', 'class', 'rating',
            ]
        );
    }

    #[Test]
    public function extract() : void
    {
        $expected = [
            ['attitude' => 'nice', 'texture' => 'furry', 'class' => 'not monster', 'rating' => '4'],
            ['attitude' => 'mean', 'texture' => 'furry', 'class' => 'monster', 'rating' => '-1.5'],
            ['attitude' => 'nice', 'texture' => 'rough', 'class' => 'not monster', 'rating' => '2.6'],
            ['attitude' => 'mean', 'texture' => 'rough', 'class' => 'monster', 'rating' => '-1'],
            ['attitude' => 'nice', 'texture' => 'rough', 'class' => 'not monster', 'rating' => '2.9'],
            ['attitude' => 'nice', 'texture' => 'furry', 'class' => 'not monster', 'rating' => '-5'],
        ];

        $records = iterator_to_array($this->extractor, false);

        $this->assertEquals($expected, $records);
    }

    #[Test]
    public function extractWithIntegerKeys() : void
    {
        $iterable = (function () {
            yield [0 => 'nice', 1 => 'furry', 2 => 'not monster', 3 => '4'];
            yield [0 => 'mean', 1 => 'furry', 2 => 'monster', 3 => '-1.5'];
        })();

        $extractor = new ColumnPicker($iterable, [2, 0, 1, 3]);

        $expected = [
            [0 => 'not monster', 1 => 'nice', 2 => 'furry', 3 => '4'],
            [0 => 'monster', 1 => 'mean', 2 => 'furry', 3 => '-1.5'],
        ];

        $records = iterator_to_array($extractor, false);

        $this->assertEquals($expected, $records);
        $this->assertTrue(array_is_list($records[0]));
    }

    #[Test]
    public function extractWithNonContiguousIntegerKeys() : void
    {
        $iterable = (function () {
            yield [0 => 'a', 1 => 'b', 2 => 'c', 3 => 'd'];
        })();

        $extractor = new ColumnPicker($iterable, [0, 3]);

        $expected = [
            [0 => 'a', 1 => 'd'],
        ];

        $records = iterator_to_array($extractor, false);

        $this->assertEquals($expected, $records);
        $this->assertTrue(array_is_list($records[0]));
    }

    #[Test]
    public function extractNullColumn() : void
    {
        $iterable = (function () {
            yield [
                'attitude' => 'nice', 'texture' => null, 'class' => 'not monster', 'rating' => '4',
            ];
        })();

        $extractor = new ColumnPicker($iterable, ['texture']);

        $expected = [
            ['texture' => null],
        ];

        $records = iterator_to_array($extractor, false);

        $this->assertEquals($expected, $records);
    }
}
