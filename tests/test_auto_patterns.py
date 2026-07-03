"""
Comprehensive pytest suite for ilovetools.utils.patterns

Covers all 8 design-pattern utilities and their edge cases:
Singleton, Observer, Strategy, ChainOfResponsibility,
Command, CommandHistory, Registry, Pipeline, Builder.
"""

import threading
import pytest

from ilovetools.utils.patterns import (
    Singleton,
    Observer,
    Strategy,
    ChainOfResponsibility,
    Command,
    CommandHistory,
    Registry,
    Pipeline,
    Builder,
)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
class TestSingleton:
    """Tests for the Singleton metaclass."""

    def test_same_instance(self):
        class Foo(metaclass=Singleton):
            def __init__(self):
                self.val = 0

        a = Foo()
        b = Foo()
        assert a is b

    def test_state_shared(self):
        class Bar(metaclass=Singleton):
            def __init__(self):
                self.counter = 0

        Singleton._clear_instance(Bar)
        x = Bar()
        x.counter = 42
        y = Bar()
        assert y.counter == 42

    def test_clear_instance(self):
        class Baz(metaclass=Singleton):
            def __init__(self):
                self.data = "first"

        Singleton._clear_instance(Baz)
        obj1 = Baz()
        obj1.data = "modified"
        Singleton._clear_instance(Baz)
        obj2 = Baz()
        assert obj2.data == "first"
        assert obj1 is not obj2

    def test_thread_safety(self):
        class ThreadSafe(metaclass=Singleton):
            def __init__(self):
                self.id = id(self)

        Singleton._clear_instance(ThreadSafe)
        instances = []

        def create():
            instances.append(ThreadSafe())

        threads = [threading.Thread(target=create) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first = instances[0]
        assert all(inst is first for inst in instances)

    def test_different_classes_different_instances(self):
        class A(metaclass=Singleton):
            pass

        class B(metaclass=Singleton):
            pass

        assert A() is not B()


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------
class TestObserver:
    """Tests for the Observer event bus."""

    def test_subscribe_and_publish(self):
        bus = Observer()
        received = []
        bus.subscribe("test", lambda d: received.append(d))
        bus.publish("test", "hello")
        assert received == ["hello"]

    def test_multiple_subscribers(self):
        bus = Observer()
        results = []
        bus.subscribe("evt", lambda d: results.append(("a", d)))
        bus.subscribe("evt", lambda d: results.append(("b", d)))
        bus.publish("evt", 99)
        assert results == [("a", 99), ("b", 99)]

    def test_unsubscribe(self):
        bus = Observer()
        sid = bus.subscribe("evt", lambda d: None)
        assert bus.subscriber_count("evt") == 1
        assert bus.unsubscribe("evt", sid) is True
        assert bus.subscriber_count("evt") == 0

    def test_unsubscribe_nonexistent(self):
        bus = Observer()
        assert bus.unsubscribe("nope", 999) is False

    def test_publish_no_subscribers(self):
        bus = Observer()
        results = bus.publish("ghost", "data")
        assert results == []

    def test_subscriber_count(self):
        bus = Observer()
        bus.subscribe("a", lambda d: None)
        bus.subscribe("a", lambda d: None)
        bus.subscribe("b", lambda d: None)
        assert bus.subscriber_count("a") == 2
        assert bus.subscriber_count("b") == 1
        assert bus.subscriber_count("c") == 0

    def test_clear(self):
        bus = Observer()
        bus.subscribe("x", lambda d: None)
        bus.subscribe("y", lambda d: None)
        bus.clear()
        assert bus.subscriber_count("x") == 0
        assert bus.subscriber_count("y") == 0

    def test_callback_exception_does_not_break_others(self):
        bus = Observer()
        good_results = []
        bus.subscribe("evt", lambda d: (_ for _ in ()).throw(ValueError("boom")))
        bus.subscribe("evt", lambda d: good_results.append(d))
        bus.publish("evt", 42)
        assert good_results == [42]

    def test_publish_returns_results(self):
        bus = Observer()
        bus.subscribe("evt", lambda d: d * 2)
        bus.subscribe("evt", lambda d: d + 1)
        results = bus.publish("evt", 10)
        assert results == [20, 11]

    def test_publish_with_kwargs(self):
        bus = Observer()
        captured = {}
        bus.subscribe("evt", lambda **kw: captured.update(kw))
        bus.publish("evt", key="value", num=42)
        assert captured == {"key": "value", "num": 42}


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class TestStrategy:
    """Tests for the Strategy context."""

    def test_register_and_execute(self):
        s = Strategy()
        s.register("add", lambda a, b: a + b)
        assert s.execute("add", 3, 4) == 7

    def test_default_strategy(self):
        s = Strategy(default="upper")
        s.register("upper", lambda t: t.upper())
        s.register("lower", lambda t: t.lower())
        assert s.execute(None, "hello") == "HELLO"

    def test_change_default(self):
        s = Strategy(default="a")
        s.register("a", lambda: "A")
        s.register("b", lambda: "B")
        s.default = "b"
        assert s.execute() == "B"

    def test_set_default_to_unregistered_raises(self):
        s = Strategy()
        with pytest.raises(KeyError):
            s.default = "ghost"

    def test_execute_unregistered_raises(self):
        s = Strategy()
        with pytest.raises(KeyError):
            s.execute("ghost")

    def test_execute_no_default_no_name_raises(self):
        s = Strategy()
        with pytest.raises(KeyError):
            s.execute()

    def test_unregister(self):
        s = Strategy()
        s.register("x", lambda: 1)
        assert s.unregister("x") is True
        assert "x" not in s.available

    def test_unregister_default_clears_default(self):
        s = Strategy(default="x")
        s.register("x", lambda: 1)
        s.unregister("x")
        assert s.default is None

    def test_register_empty_name_raises(self):
        s = Strategy()
        with pytest.raises(ValueError):
            s.register("", lambda: 1)

    def test_available(self):
        s = Strategy()
        s.register("c", lambda: 1)
        s.register("a", lambda: 2)
        s.register("b", lambda: 3)
        assert s.available == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# ChainOfResponsibility
# ---------------------------------------------------------------------------
class TestChainOfResponsibility:
    """Tests for the Chain of Responsibility."""

    def test_first_match_wins(self):
        chain = ChainOfResponsibility()
        chain.add(lambda r: r["type"] == "A", lambda r: "handled-A")
        chain.add(lambda r: r["type"] == "B", lambda r: "handled-B")
        assert chain.handle({"type": "A"}) == "handled-A"
        assert chain.handle({"type": "B"}) == "handled-B"

    def test_no_match_returns_default(self):
        chain = ChainOfResponsibility(default="fallback")
        chain.add(lambda r: r["type"] == "A", lambda r: "handled-A")
        assert chain.handle({"type": "Z"}) == "fallback"

    def test_no_match_default_none(self):
        chain = ChainOfResponsibility()
        chain.add(lambda r: r["type"] == "A", lambda r: "handled-A")
        assert chain.handle({"type": "Z"}) is None

    def test_chaining_add(self):
        chain = ChainOfResponsibility()
        ret = chain.add(lambda r: True, lambda r: "ok")
        assert ret is chain

    def test_reset(self):
        chain = ChainOfResponsibility()
        chain.add(lambda r: True, lambda r: "ok")
        chain.reset()
        assert chain.length == 0

    def test_length(self):
        chain = ChainOfResponsibility()
        chain.add(lambda r: True, lambda r: 1)
        chain.add(lambda r: False, lambda r: 2)
        assert chain.length == 2

    def test_empty_chain(self):
        chain = ChainOfResponsibility(default="empty")
        assert chain.handle({}) == "empty"

    def test_order_matters(self):
        chain = ChainOfResponsibility()
        chain.add(lambda r: r["n"] > 0, lambda r: "positive")
        chain.add(lambda r: r["n"] > 10, lambda r: "big")
        # First matching handler wins, so 20 hits "positive" not "big"
        assert chain.handle({"n": 20}) == "positive"


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------
class TestCommand:
    """Tests for the Command pattern."""

    def test_execute(self):
        state = {"v": 0}
        cmd = Command(
            do=lambda: state.__setitem__("v", 1),
            undo=lambda: state.__setitem__("v", 0),
        )
        cmd.execute()
        assert state["v"] == 1
        assert cmd.is_executed is True

    def test_undo(self):
        state = {"v": 10}
        cmd = Command(
            do=lambda: state.__setitem__("v", 20),
            undo=lambda: state.__setitem__("v", 10),
        )
        cmd.execute()
        cmd.undo()
        assert state["v"] == 10
        assert cmd.is_executed is False

    def test_redo(self):
        state = {"v": 0}
        cmd = Command(
            do=lambda: state.__setitem__("v", 5),
            undo=lambda: state.__setitem__("v", 0),
        )
        cmd.execute()
        cmd.undo()
        cmd.redo()
        assert state["v"] == 5
        assert cmd.is_executed is True

    def test_undo_without_callback_raises(self):
        cmd = Command(do=lambda: None)
        cmd.execute()
        with pytest.raises(RuntimeError):
            cmd.undo()

    def test_undo_before_execute_raises(self):
        cmd = Command(do=lambda: None, undo=lambda: None)
        with pytest.raises(RuntimeError):
            cmd.undo()

    def test_name(self):
        cmd = Command(do=lambda: None, name="my-command")
        assert cmd.name == "my-command"


class TestCommandHistory:
    """Tests for the CommandHistory manager."""

    def test_execute_and_undo(self):
        state = {"n": 0}
        history = CommandHistory()
        history.execute(Command(
            do=lambda: state.__setitem__("n", state["n"] + 1),
            undo=lambda: state.__setitem__("n", state["n"] - 1),
        ))
        assert state["n"] == 1
        history.undo()
        assert state["n"] == 0

    def test_redo_after_undo(self):
        state = {"n": 0}
        history = CommandHistory()
        history.execute(Command(
            do=lambda: state.__setitem__("n", state["n"] + 5),
            undo=lambda: state.__setitem__("n", state["n"] - 5),
        ))
        history.undo()
        history.redo()
        assert state["n"] == 5

    def test_multiple_undo(self):
        state = {"n": 0}
        history = CommandHistory()
        for i in range(3):
            history.execute(Command(
                do=lambda: state.__setitem__("n", state["n"] + 1),
                undo=lambda: state.__setitem__("n", state["n"] - 1),
            ))
        assert state["n"] == 3
        history.undo()
        assert state["n"] == 2
        history.undo()
        assert state["n"] == 1

    def test_undo_empty_raises(self):
        history = CommandHistory()
        with pytest.raises(RuntimeError):
            history.undo()

    def test_redo_empty_raises(self):
        history = CommandHistory()
        with pytest.raises(RuntimeError):
            history.redo()

    def test_can_undo_can_redo(self):
        history = CommandHistory()
        assert history.can_undo is False
        assert history.can_redo is False
        history.execute(Command(do=lambda: None, undo=lambda: None))
        assert history.can_undo is True
        history.undo()
        assert history.can_redo is True
        assert history.can_undo is False

    def test_new_execute_clears_redo(self):
        state = {"n": 0}
        history = CommandHistory()
        history.execute(Command(
            do=lambda: state.__setitem__("n", state["n"] + 1),
            undo=lambda: state.__setitem__("n", state["n"] - 1),
        ))
        history.undo()
        assert history.can_redo is True
        history.execute(Command(
            do=lambda: state.__setitem__("n", state["n"] + 10),
            undo=lambda: state.__setitem__("n", state["n"] - 10),
        ))
        assert history.can_redo is False

    def test_clear(self):
        history = CommandHistory()
        history.execute(Command(do=lambda: None, undo=lambda: None))
        history.clear()
        assert history.can_undo is False
        assert history.can_redo is False

    def test_max_size(self):
        history = CommandHistory(max_size=3)
        for _ in range(5):
            history.execute(Command(do=lambda: None, undo=lambda: None))
        # Only the last 3 should be on the undo stack
        history.undo()
        history.undo()
        history.undo()
        with pytest.raises(RuntimeError):
            history.undo()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestRegistry:
    """Tests for the Registry pattern."""

    def test_register_decorator(self):
        reg = Registry()

        @reg.register("greet")
        def greet(name):
            return f"Hello, {name}!"

        assert reg.get("greet")("world") == "Hello, world!"

    def test_register_class(self):
        reg = Registry()

        @reg.register("dog")
        class Dog:
            def speak(self):
                return "Woof"

        assert reg.get("dog")().speak() == "Woof"

    def test_register_instance(self):
        reg = Registry()
        reg.register_instance("config", {"debug": True})
        assert reg.get("config")["debug"] is True

    def test_duplicate_raises(self):
        reg = Registry()
        reg.register_instance("x", 1)
        with pytest.raises(ValueError):
            reg.register_instance("x", 2)

    def test_duplicate_decorator_raises(self):
        reg = Registry()

        @reg.register("x")
        def f1():
            pass

        with pytest.raises(ValueError):

            @reg.register("x")
            def f2():
                pass

    def test_get_missing_raises(self):
        reg = Registry()
        with pytest.raises(KeyError):
            reg.get("ghost")

    def test_unregister(self):
        reg = Registry()
        reg.register_instance("x", 1)
        assert reg.unregister("x") is True
        assert "x" not in reg
        assert reg.unregister("x") is False

    def test_names(self):
        reg = Registry()
        reg.register_instance("c", 1)
        reg.register_instance("a", 2)
        reg.register_instance("b", 3)
        assert reg.names == ["a", "b", "c"]

    def test_contains(self):
        reg = Registry()
        reg.register_instance("x", 1)
        assert "x" in reg
        assert "y" not in reg

    def test_len(self):
        reg = Registry()
        reg.register_instance("a", 1)
        reg.register_instance("b", 2)
        assert len(reg) == 2

    def test_iter(self):
        reg = Registry()
        reg.register_instance("a", 1)
        reg.register_instance("b", 2)
        assert sorted(reg) == ["a", "b"]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class TestPipeline:
    """Tests for the Pipeline pattern."""

    def test_basic_pipeline(self):
        pipe = Pipeline()
        pipe.add(lambda x: x + 1)
        pipe.add(lambda x: x * 2)
        assert pipe.run(10) == 22

    def test_named_stages(self):
        pipe = Pipeline()
        pipe.add(lambda s: s.strip(), name="strip")
        pipe.add(lambda s: s.upper(), name="upper")
        assert pipe.run("  hello  ") == "HELLO"
        assert pipe.stage_names == ["strip", "upper"]

    def test_chaining(self):
        pipe = Pipeline()
        ret = pipe.add(lambda x: x)
        assert ret is pipe

    def test_empty_pipeline_raises(self):
        pipe = Pipeline()
        with pytest.raises(RuntimeError):
            pipe.run(42)

    def test_length(self):
        pipe = Pipeline()
        pipe.add(lambda x: x + 1)
        pipe.add(lambda x: x + 2)
        pipe.add(lambda x: x + 3)
        assert pipe.length == 3

    def test_reset(self):
        pipe = Pipeline()
        pipe.add(lambda x: x + 1)
        pipe.reset()
        assert pipe.length == 0

    def test_anonymous_stage_name(self):
        pipe = Pipeline()
        pipe.add(lambda x: x + 1)
        assert pipe.stage_names == ["<anonymous>"]

    def test_data_transformation_chain(self):
        pipe = Pipeline()
        pipe.add(lambda data: {**data, "step1": True})
        pipe.add(lambda data: {**data, "step2": True})
        pipe.add(lambda data: {**data, "step3": True})
        result = pipe.run({"initial": 1})
        assert result == {"initial": 1, "step1": True, "step2": True, "step3": True}

    def test_string_pipeline(self):
        pipe = Pipeline()
        pipe.add(lambda s: s.replace("world", "Python"))
        pipe.add(lambda s: s.title())
        pipe.add(lambda s: s + "!")
        assert pipe.run("hello world") == "Hello Python!"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class TestBuilder:
    """Tests for the Builder pattern."""

    def test_basic_build(self):
        class User:
            def __init__(self, name="", email=""):
                self.name = name
                self.email = email

        user = Builder(User).set("name", "Alice").set("email", "a@b.com").build()
        assert user.name == "Alice"
        assert user.email == "a@b.com"

    def test_set_many(self):
        class Config:
            def __init__(self, debug=False, verbose=False, port=8080):
                self.debug = debug
                self.verbose = verbose
                self.port = port

        cfg = Builder(Config).set_many(debug=True, port=3000).build()
        assert cfg.debug is True
        assert cfg.verbose is False
        assert cfg.port == 3000

    def test_chaining(self):
        class Item:
            def __init__(self, a=0, b=0, c=0):
                self.a = a
                self.b = b
                self.c = c

        b = Builder(Item)
        ret = b.set("a", 1)
        assert ret is b
        item = b.set("b", 2).set("c", 3).build()
        assert (item.a, item.b, item.c) == (1, 2, 3)

    def test_reset(self):
        class User:
            def __init__(self, name=""):
                self.name = name

        b = Builder(User).set("name", "Bob")
        b.reset()
        user = b.build()
        assert user.name == ""

    def test_empty_build(self):
        class Empty:
            def __init__(self):
                pass

        obj = Builder(Empty).build()
        assert isinstance(obj, Empty)

    def test_attributes(self):
        class User:
            def __init__(self, name="", age=0):
                self.name = name
                self.age = age

        b = Builder(User).set("name", "Alice").set("age", 30)
        attrs = b.attributes
        assert attrs == {"name": "Alice", "age": 30}
        # Ensure it's a copy
        attrs["name"] = "HACKED"
        assert b.attributes["name"] == "Alice"

    def test_factory_function(self):
        def create_point(x=0, y=0):
            return (x, y)

        point = Builder(create_point).set("x", 5).set("y", 10).build()
        assert point == (5, 10)

    def test_override_value(self):
        class User:
            def __init__(self, name=""):
                self.name = name

        user = Builder(User).set("name", "First").set("name", "Second").build()
        assert user.name == "Second"
