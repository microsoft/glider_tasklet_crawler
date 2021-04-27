class YlWindowState {
	constructor() {
        this.bodyState = YlDomState.extractFrom(document.body);
        if (this.bodyState) {
            this.elements = this.bodyState.traverse();
        }
    }

	toJSON() {
        return JSON.stringify({
            host: window.location.host,
            url: window.location.href,
            title: document.title,
            scrollX: window.scrollX,
            scrollY: window.scrollY,
            windowWidth: window.innerWidth,
            windowHeight: window.innerHeight,
            elements: this.elements
        });
    }
}

// YlDomState is the state related to a DOM element
class YlDomState {
	constructor(domElement) {
		this.domElement = domElement;

		// TODO improve feature extraction in the future
		this.bound = domElement.getYlBound();
		this.style = domElement.getYlStyle();
        this.text = domElement.getYlText();
		this.actionSet = domElement.getYlActionSet();

		this.isVisible = false;
		if (!this.hidden && this.style.visibility != 'hidden' && this.style.display != 'none' &&
		    !("getAttribute" in this && this.getAttribute("aria-hidden") == true) &&
            this.bound && this.bound.top < this.bound.bottom && this.bound.left < this.bound.right)
		    this.isVisible = true;

		this.parent = null;
		this.children = [];
	}

	traverse() {
		var eleStates = [];
		eleStates.push(this);
		for (var i in this.children) {
		    if (this.children[i] && this.children[i].traverse)
			    eleStates = eleStates.concat(this.children[i].traverse());
		}
		return eleStates;
	}

	static extractFrom(domElement) {
		if (!domElement || domElement.nodeType != 1 || domElement.disabled || !domElement.getYlBound) return null;
		// console.log(domElement);
		var eleState = new YlDomState(domElement);

		var childElements = domElement.children;
		for (var i in childElements) {
			var childElement = childElements[i];
			var childState = YlDomState.extractFrom(childElement);
			if (childState) {
				childState.parent = eleState;
				eleState.children.push(childState);
			}
		}

        if (eleState.children.length > 0) {
            eleState.isVisible = true;
        }

        if (!eleState.isVisible) return null;
		return eleState;
	}

	toJSON() {
	    var parentElementId = -1;
        if (this.parent) parentElementId = this.parent.domElement.WebBotID;
        var childElementIds = [];
        for (var i in this.children) {
            if (this.children[i] && this.children[i].domElement)
                childElementIds.push(this.children[i].domElement.WebBotID);
        }

        return {
            WebBotID: this.domElement.WebBotID,
            xpath: this.domElement.ylXPath,
            xpathShort: this.domElement.ylXPathShort,
            locator: this.domElement.ylLocator,
            tagName: this.domElement.tagName,
            domId: this.domElement.ylDomId,
            class: this.domElement.className,
            title: this.domElement.title,
            value: this.domElement.value,
            placeholder: this.domElement.placeholder,
            type: this.domElement.type,
            ariaLabel: this.domElement.getAttribute("aria-label"),
            labelText: this.domElement.getYlLabelText(),
            labelValue: this.domElement.getYlLabelValue(),

            bound: this.bound,
            style: this.style,
            actionSet: this.actionSet,
            text: this.text,
            parent: parentElementId,
            children: childElementIds
        }
	}
}

class YlWindowController {
    constructor() {
        this.timeout = null;
        this.state = null;
        this.stateTime = 0;
        this.actionLog = [];
    }

    update() {
        if (document.hidden) return; // Do not update while the current window is in background

        var millis = Date.now() - this.stateTime;
        if (millis < 100) return;

        var state = this.getCurrentState();
        if (!state) return;
        console.log(state);

        this.state = state;
        this.stateTime = Date.now();
        state.requestAction();
    }

    start() {
        var outer = this;
        MutationObserver = window.MutationObserver || window.WebKitMutationObserver;
        this.mutationObserver = new MutationObserver(function(mutations, observer) {
            // fired when a mutation occurs
            // console.log(mutations, observer);
            if (outer.timeout) clearTimeout(outer.timeout);
            outer.timeout = setTimeout(outer.update, 100);
        });

        // define what element should be observed by the observer
        // and what types of mutations trigger the callback
        this.mutationObserver.observe(document, {
            subtree: true,
            childList: true,
            attributes: true,
            attributeOldValue: true,
            characterData: true,
            characterOldData: true,
            attributeFilter: ["id", "style", "value"]
        });

        // periodically update current window
        window.setInterval(this.update, 1000)
    }

	indexAllElements() {
	    var elements = document.getElementsByTagName("*");
	    var id2count = new Map();
	    for (var i in elements) {
	        var ele = elements[i];
	        if (!ele || ele.nodeType != 1 || ele.disabled) continue;
	        ele.WebBotID = i;
	        ele.setAttribute('webbotid', i);
	        if ('id' in ele) {
	            if (id2count.has(ele.id)) id2count.set(ele.id, id2count.get(ele.id) + 1);
	            else id2count.set(ele.id, 1);
	        }
	    }

	    for (var i in elements) {
    	    var ele = elements[i];
	        if (!ele || ele.nodeType != 1 || ele.disabled || !ele.createYlLocator) continue;
	        ele.createYlLocator(id2count);
	    }
    }

    getCurrentState() {
	    if (document.body) {
            this.indexAllElements();
		    var windowState = new YlWindowState();
		    if (windowState.bodyState && windowState.elements) {
		        return windowState;
		    }
	    }
	}

	getInteractableElements() {
	    var elements = document.getElementsByTagName("*");
	    var interactableElements = [];
	    for (var i in elements) {
	        var ele = elements[i];
	        if (!ele || ele.nodeType != 1 || ele.disabled) continue;
	        if ('getYlActionSet' in ele) {
                var actionSet = ele.getYlActionSet();
                if (actionSet && actionSet.actionType) {
                    if (actionSet.actionType == 'select') {
                        interactableElements.push(...ele.options);
                    }
                    else {
                        interactableElements.push(ele);
                    }
                }
	        }
	    }
	    return interactableElements;
    }

    getStateJson() {
        var state = this.getCurrentState();
        if (!state) return "";
        // console.log(state);
        this.state = state;
        this.stateTime = Date.now();
        return state.toJSON();
    }

    logAction(msg) {
        console.log("WebBot " + msg);
        var xhr = new XMLHttpRequest();
	    var outer = this;
        xhr.onreadystatechange = function() {};
        xhr.open("POST", "http://localhost:7336/", true);
        xhr.setRequestHeader("Content-type", "text/plain");
        xhr.send(msg);
    }

    listenActions() {
        this.indexAllElements();
        if (document.ylListening) return;
        var controller = this;
        document.addEventListener('keydown', function(e){
            if (e.shiftKey && e.key == 'S') controller.logAction('Shift+S');
            else if (e.shiftKey && e.key == 'C') controller.logAction('Shift+C');
            else if (e.shiftKey && e.key == 'N') controller.logAction('Shift+N');
            else if (e.shiftKey && e.key == 'Q') controller.logAction('Shift+Q');
            else if (e.key == 'Enter') controller.logAction('press_enter ## @ ' + e.target.getYlLocator());
        });
        document.addEventListener('keyup', function(e){
            if (!e.ctrlKey && !e.altKey && !e.shiftKey && e.key != 'Enter' && 'value' in e.target)
                controller.logAction('input_text #' + e.target.value + '# @ ' + e.target.getYlLocator());
        });
        document.addEventListener('mouseup', function(e){
            if (e.target.tagName == 'OPTION' || e.target.tagName == 'SELECT') return;
            var clickedEle = e.target.getYlClickableParent();
            if (clickedEle) controller.logAction('click ## @ ' + clickedEle.getYlLocator());
            controller.logAction('finish ## @ ' + e.target.getYlLocator());
        });
        document.addEventListener('change', function(e){
            if (e.target.tagName == 'SELECT' && e.target.selectedOptions.length > 0) {
                controller.logAction('select #' + e.target.selectedIndex + '# @ ' + e.target.getYlLocator());
            } else if (e.target.tagName == 'INPUT' && (e.target.type == 'checkbox' || e.target.type == 'radio')) {
                controller.logAction('click ## @ ' + e.target.getYlLocator());
            }
        });
        document.ylListening = true;
    }

    findElementByXPath(path) {
        var evaluator = new XPathEvaluator();
        var result = evaluator.evaluate(path, document.documentElement, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
        return result.singleNodeValue;
    }

    requestAction() {
        var state = this.getCurrentState();
	    var xhr = new XMLHttpRequest();
	    var outer = this;
        xhr.onreadystatechange = function() {
            if (this.readyState == 4 && this.status == 200) {
                console.log(this.responseText);
                var action = JSON.parse(this.responseText);
                outer.performAction(action);
            }
        };
        xhr.open("POST", "http://localhost:8000/", true);
        xhr.setRequestHeader("Content-type", "application/json; charset=UTF-8");
        xhr.send(state.toJSON());
	}

	performAction(action) {
	    var actionType = action.action_type;
	    var targetLocator = action.target_locator;
	    var actionValue = action.value;

	    if (!actionType) {
	        console.log("Action type is invalid " + actionType)
	    } else {
	        console.log("Action: " + action);
	    }

        var ele = this.findElementByXPath(targetLocator)
        if (ele) {
            if (actionType == "click") {
                ele.click()
            } else if (actionType == "check") {
                ele.checked = true;
            } else if (actionType == "select") {
                ele.selectedIndex = actionValue;
            } else if (actionType == "setValue") {
                ele.value = actionValue;
            }
        } else {
            console.log("Action target not found.")
        }
	}

    performActionJson(actionJson) {
        var action = JSON.parse(actionJson);
        this.performAction(action);
    }
}

var setupYlHelpers = function() {
    window.ylController = new YlWindowController();
    // ylController.start();

    window.alert = function alert(msg) {
        console.log("Hidden Alert " + msg);
    };

    window.print = function print() {
        console.log("Ignored Print");
    };

    window.confirm = function confirm(msg) {
        console.log("Hidden Confirm " + msg);
        return true; /*simulates user clicking yes*/
    };

    windowOpen = window.open;
    window.open = function open(url, name, params) {
        return windowOpen(url, "_blank", params);
    }

    Element.prototype.createYlLocator = function(id2count) {
	    if ('ylLocator' in this) return;
	    if (this.parentElement && this.parentElement.createYlLocator) this.parentElement.createYlLocator(id2count);

        var parentXPath = '', parentXPathShort = '';
        if (this.parentElement && 'ylLocator' in this.parentElement) {
            parentXPath = this.parentElement.ylXPath || '';
	        parentXPathShort = this.parentElement.ylXPathShort || '';
        }

	    for (var i = 1, sib = this.previousSibling; sib; sib = sib.previousSibling) {
            if (sib.localName == this.localName)  i++;
        };
        var elePosition = this.localName.toLowerCase() + '[' + i + ']';

        this.ylXPath = parentXPath + "/" + elePosition;
	    if (this.id && id2count.get(this.id) == 1) {
	        this.ylXPathShort = 'id("' + this.id + '")';
	        this.ylDomId = this.id;
	    } else {
	        this.ylXPathShort = parentXPathShort + "/" + elePosition;
	    }

	    var domText = '';
        if (this.text && this.text.substring) domText = this.text.substring(0, 10);
        // var textXPath = '//' + this.localName + '[contains(text(),"' + domText + '")]'
        this.ylLocator = this.ylXPathShort + " || " + this.ylXPath;
    }

    Element.prototype.getYlLocator = function() {
        if ('ylLocator' in this) return this.ylLocator;
        window.ylController.indexAllElements();
        return this.ylLocator;
    }

    Element.prototype.getYlLabelValue = function() {
        if (this.tagName == "LABEL" && this.hasAttribute("for")) {
            var forId = this.getAttribute("for");
            var forEle = document.getElementById(forId);
            if (forEle && "value" in forEle) {
                return forEle.value;
            }
        }
    }

    Element.prototype.getYlLabelText = function() {
        if ("id" in this) {
            var thisId = this.id;
            var labelEles = document.getElementsByTagName("LABEL");
            for (var i in labelEles) {
                var labelEle = labelEles[i];
                if (labelEle.nodeType == 1 && labelEle.hasAttribute("for") && (labelEle.getAttribute("for") == thisId) && "textContent" in labelEle) {
                    return labelEle.textContent;
                }
            }
        }
    }

    Element.prototype.getYlText = function() {
        if (this.tagName == "SELECT") {
            var text = "";
            for (var i in this.selectedOptions) {
                var selectedText = this.selectedOptions[i].text;
                if (selectedText)
                    text = text + " " + selectedText;
            }
            return text.trim();
        }
        var childNodes = this.childNodes;
        var text = "";
        var inlineTags = ["a", "abbr", "acronym", "b", "bdi", "bdo", "big", "cite", "code", "data", "del", "dfn",
            "em", "i", "ins", "mark", "meter", "s", "span", "strong", "sub", "sup", "time", "u"];
        for (var i in childNodes) {
            var childNode = childNodes[i];
            if (childNode.nodeType == 3) {
                // If the child node is a text node
                var nodeValue = childNode.nodeValue;
                if (nodeValue) text = text + " " + nodeValue;
            } else if (childNode.nodeType == 1 && text && inlineTags.includes(childNode.localName.toLowerCase())) {
                var inlineNodeText = childNode.innerText;
                if (inlineNodeText) text = text + " " + inlineNodeText;
            }
        }
        return text.trim();
    }

    Element.prototype.getYlBound = function() {
        if (!this.getBoundingClientRect) return null;
        var eleRect = this.getBoundingClientRect();
        return {
            left: eleRect.left,
            right: eleRect.right,
            top: eleRect.top,
            bottom: eleRect.bottom
        }
    }

    Element.prototype.getYlStyle = function() {
        var computedStyle = window.getComputedStyle(this);
        return {
            fontSize: parseFloat(computedStyle.fontSize) || 0,
            fontWeight: parseFloat(computedStyle.fontWeight) || 0,
            color: computedStyle.color,
            cursor: computedStyle.cursor,
            zIndex: computedStyle.zIndex,
            visibility: computedStyle.visibility,
            display: computedStyle.display,
            bgColor: computedStyle.backgroundColor,
            hasBgImg: computedStyle.backgroundImage != "none",
            hasBorder: computedStyle.borderWidth != "0px" && computedStyle.borderStyle != "none"
        }
    }

    Element.prototype.getYlActionSet = function() {
        var tag = this.tagName;
        var actionSet = {};
        if (this.form) actionSet.formId = this.form.WebBotID;
        if (tag == "A") {
            actionSet.actionType = "click";
            actionSet.href = this.href;
        } else if (tag == "BUTTON") {
            actionSet.actionType = "click";
        } else if (tag == "INPUT") {
            var inputType = this.type;
            if (inputType == "button" || inputType == "submit" || inputType == "reset" || inputType == "image") {
                actionSet.actionType = "click";
            } else if (inputType == "checkbox" || inputType == "radio") {
                actionSet.actionType = "check";
                actionSet.checked = this.checked;
                actionSet.defaultChecked = this.defaultChecked;
            } else if (inputType == "file" || inputType == "hidden" || this.hasAttribute("readonly")) {
                // Ignore file input as it will open an external window
                actionSet = null;
            } else {
                actionSet.actionType = "setValue";
                actionSet.defaultValue = this.defaultValue;
            }
        } else if (tag == "TEXTAREA" && !this.hasAttribute("readonly")) {
            actionSet.actionType = "setValue";
        } else if (tag == "SELECT") {
            actionSet.actionType = "select";
            actionSet.multiple = this.parentElement.multiple;
            actionSet.selectedIndices = []
            actionSet.options = []
            for (var i = 0; i < this.options.length; i++) {
                var option = this.options[i];
                actionSet.options.push(option.text);
                if (option.selected) {
                    actionSet.selectedIndices.push(i);
                }
            }
        } else if (this.onclick || this.onmouseup || this.onmousedown || this.hasAttribute("click.trigger")) {
            actionSet.actionType = "click";
        } else if (this.getYlStyle().cursor == "pointer" &&
                   this.parentElement &&
                   this.parentElement.getYlStyle().cursor != "pointer") {
            actionSet.actionType = "click";
//        } else if (this.getAttribute("role") == "gridcell") {
//            actionSet.actionType = "click";
        } else {
            actionSet = null;
        }
        // Note: there might some elements with an onclick listener, but we ignore them for now
        return actionSet;
    }

    Element.prototype.getYlClickableParent = function() {
        actionSet = this.getYlActionSet();
        if (actionSet && actionSet.actionType == "click") return this;
        if (this.tagName == "INPUT") return this; // TODO: better way to decide whether clickable
        if (this.parentElement) return this.parentElement.getYlClickableParent();
    }
}

//document.body.style.zoom=0.5;
setupYlHelpers();
