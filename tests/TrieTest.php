<?php

use Rubix\Engine\Trie;
use Rubix\Engine\Node;
use PHPUnit\Framework\TestCase;

class TrieTest extends TestCase
{
    protected $trie;

    public function setUp()
    {
        $this->trie = new Trie([
            'its', 'just', 'literally', 'a', 'normal', 'car', 'in', 'space', 'i', 'like', 'the',
            'absurdity', 'of', 'that', 'its', 'silly', 'and', 'fun', 'but', 'i', 'think', 'that',
            'fun', 'silly', 'things', 'are', 'important', 'stay', 'calm']);
    }

    public function test_build_trie()
    {
        $this->assertTrue($this->trie instanceof Trie);
        $this->assertTrue($this->trie->root() instanceof Node);
    }

    public function test_has_word()
    {
        $this->assertTrue($this->trie->has('literally'));
        $this->assertTrue($this->trie->has('important'));
        $this->assertFalse($this->trie->has('earth'));
        $this->assertFalse($this->trie->has('mars'));
    }

    public function test_insert_word()
    {
        $node = $this->trie->insert('hairy');

        $this->assertTrue($node instanceof Node);
        $this->assertEquals('y', $node->id());
        $this->assertEquals('r', $node->parent->id());
        $this->assertEquals('i', $node->parent->parent->id());
        $this->assertEquals('a', $node->parent->parent->parent->id());
        $this->assertEquals('h', $node->parent->parent->parent->parent->id());
        $this->assertTrue($node->word);
        $this->assertNull($node->parent->word);
        $this->assertNull($node->parent->parent->word);
        $this->assertNull($node->parent->parent->parent->word);
        $this->assertNull($node->parent->parent->parent->parent->word);
    }

    public function test_find_prefix()
    {
        $node = $this->trie->find('norma');

        $this->assertTrue($node instanceof Node);
        $this->assertEquals('a', $node->id());
        $this->assertEquals('m', $node->parent->id());
        $this->assertEquals('r', $node->parent->parent->id());
        $this->assertEquals('o', $node->parent->parent->parent->id());
        $this->assertEquals('n', $node->parent->parent->parent->parent->id());
        $this->assertNull($node->word);
    }

    public function test_delete_word()
    {
        $this->assertTrue($this->trie->has('normal'));
        $this->assertTrue($this->trie->has('literally'));

        $trie = $this->trie->delete('normal');

        $this->assertFalse($trie->has('normal'));
        $this->assertTrue($trie->has('literally'));
    }

    public function test_size()
    {
        $this->assertEquals(24, $this->trie->size());

        $this->trie->insert('ghost');

        $this->assertEquals(25, $this->trie->size());

        $this->trie->insert('space');

        $this->assertEquals(25, $this->trie->size());

        $this->trie->delete('the');

        $this->assertEquals(24, $this->trie->size());
    }
}
